/*
 * This file is part of crest.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Modifications for MLIP support:
 * Copyright (C) 2024-2026 Alexander Kolganov
 *
 * crest is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * crest is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with crest.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * libtorch_bridge.cpp — C++ implementation of TorchScript MLIP inference.
 *
 * Supports two model formats:
 *   format=0 (generic): forward(positions, Z) -> (energy_hartree, gradient_hartree_bohr)
 *   format=1 (MACE LAMMPS): forward(data_dict, local_or_ghost) -> result_dict
 *     The MACE model operates in eV/Angstrom, so we convert units here.
 *
 * Thread safety:
 *   - Per-thread mode: Each LibtorchContext is independent.
 *   - Shared mode: SharedModelRegistry provides thread-safe access.
 *   - Batched mode: libtorch_engrad_batch() for GPU throughput.
 */

#include "libtorch_bridge.h"

#ifdef WITH_LIBTORCH

#include <torch/script.h>
#include <torch/torch.h>

#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <algorithm>

/* Unit conversion constants */
static constexpr double BOHR_TO_ANG = 0.529177210903;
static constexpr double EV_TO_HARTREE = 1.0 / 27.211386245988;
static constexpr double ANG_TO_BOHR = 1.0 / BOHR_TO_ANG;


/* ====================================================================
 * Helper functions
 * ==================================================================== */

/* Helper: write error message to buffer safely */
static void set_error(char* err_msg, int err_len, const std::string& msg) {
    if (err_msg && err_len > 0) {
        int n = static_cast<int>(msg.size());
        if (n >= err_len) n = err_len - 1;
        std::memcpy(err_msg, msg.c_str(), n);
        err_msg[n] = '\0';
    }
}

/* Map integer device code to torch::Device.
 * Encoding: 0=CPU, 1=CUDA:0 (compat), 2=MPS, 10+idx=CUDA:idx */
static torch::Device device_from_int(int device_code) {
    if (device_code >= 10 && device_code < 20) {
        return torch::Device(torch::kCUDA, device_code - 10);
    }
    switch (device_code) {
        case 1:  return torch::Device(torch::kCUDA, 0);
        case 2:
#ifdef __APPLE__
            return torch::Device("mps");
#else
            return torch::kCPU;
#endif
        default: return torch::kCPU;
    }
}


/* Internal context holding the loaded model and parameters */
struct LibtorchContext {
    torch::jit::script::Module module;
    torch::Device device;
    int model_format;    /* 0=generic, 1=MACE */
    double cutoff_ang;   /* cutoff in Angstrom for neighbor list */
    bool initialized;

    /* Cached tensors for MACE format — rebuilt only when nat/Z changes */
    int cached_nat;
    std::vector<int> cached_z;
    torch::Tensor cached_node_attrs;
    torch::Tensor cached_batch;
    torch::Tensor cached_ptr;
    torch::Tensor cached_cell;
    torch::Tensor cached_pbc;
    torch::Tensor cached_local_or_ghost;

    /* Mutex for shared model access (only used in shared mode) */
    std::mutex forward_mutex;

    LibtorchContext() : device(torch::kCPU), model_format(0),
                        cutoff_ang(6.0), initialized(false),
                        cached_nat(0) {}
};


/* ====================================================================
 * Shared Model Registry — thread-safe singleton for GPU model sharing
 * ==================================================================== */

struct SharedModelRegistry {
    std::unordered_map<std::string, std::shared_ptr<LibtorchContext>> models;
    std::mutex registry_mutex;

    LibtorchContext* get_or_load(const char* path, int device_code,
                                 int model_format, double cutoff,
                                 char* err_msg, int err_len)
    {
        /* Key includes device so same model on different GPUs gets separate entries */
        std::string key = std::string(path) + "@dev" + std::to_string(device_code);
        std::lock_guard<std::mutex> lock(registry_mutex);

        auto it = models.find(key);
        if (it != models.end() && it->second->initialized) {
            return it->second.get();
        }

        try {
            auto ctx = std::make_shared<LibtorchContext>();
            ctx->device = device_from_int(device_code);
            ctx->model_format = model_format;
            ctx->cutoff_ang = cutoff > 0 ? cutoff : 6.0;

            ctx->module = torch::jit::load(path, ctx->device);
            ctx->module.eval();
            torch::jit::setGraphExecutorOptimize(true);

            ctx->initialized = true;
            models[key] = ctx;
            return ctx.get();

        } catch (const c10::Error& e) {
            set_error(err_msg, err_len,
                      std::string("libtorch shared load error: ") + e.what());
            return nullptr;
        } catch (const std::exception& e) {
            set_error(err_msg, err_len,
                      std::string("libtorch shared load error: ") + e.what());
            return nullptr;
        }
    }

    void free_all() {
        std::lock_guard<std::mutex> lock(registry_mutex);
        models.clear();
    }
};

static SharedModelRegistry g_shared_models;


/* ====================================================================
 * Batch workspace for pipelined GPU inference
 * ==================================================================== */

struct BatchWorkspace {
    /* GPU-side tensors (set by prepare, used by launch) */
    torch::Tensor positions, edge_index, shifts;
    torch::Tensor batch_tensor, ptr_tensor, node_attrs;
    torch::Tensor local_or_ghost, cell, pbc;

    /* Forward pass output (set by launch, read by collect) */
    torch::jit::IValue output;
    bool has_output;

    /* Batch metadata */
    int bcount;   /* actual structures in this batch */
    int nat;      /* atoms per structure */

    BatchWorkspace() : has_output(false), bcount(0), nat(0) {}
};


/* ====================================================================
 * Neighbor list construction
 * ==================================================================== */

/*
 * Build neighbor list for non-periodic system using cell lists.
 * O(N * avg_neighbors) instead of O(N²).
 * Returns (edge_index[2, num_edges], shifts[num_edges, 3])
 */
static std::pair<torch::Tensor, torch::Tensor>
build_neighbor_list(const torch::Tensor& positions, double cutoff) {
    int64_t nat = positions.size(0);
    double cutoff2 = cutoff * cutoff;

    auto pos_acc = positions.accessor<double, 2>();

    /* For small systems, use simple O(N²) loop */
    if (nat <= 64) {
        std::vector<int64_t> src_vec, dst_vec;
        src_vec.reserve(nat * 20);
        dst_vec.reserve(nat * 20);

        for (int64_t i = 0; i < nat; i++) {
            for (int64_t j = 0; j < nat; j++) {
                if (i == j) continue;
                double dx = pos_acc[i][0] - pos_acc[j][0];
                double dy = pos_acc[i][1] - pos_acc[j][1];
                double dz = pos_acc[i][2] - pos_acc[j][2];
                if (dx*dx + dy*dy + dz*dz <= cutoff2) {
                    src_vec.push_back(i);
                    dst_vec.push_back(j);
                }
            }
        }

        int64_t num_edges = static_cast<int64_t>(src_vec.size());
        auto edge_index = torch::empty({2, num_edges}, torch::kInt64);
        std::memcpy(edge_index.data_ptr<int64_t>(),
                    src_vec.data(), num_edges * sizeof(int64_t));
        std::memcpy(edge_index.data_ptr<int64_t>() + num_edges,
                    dst_vec.data(), num_edges * sizeof(int64_t));
        auto shifts = torch::zeros({num_edges, 3}, torch::kFloat64);
        return {edge_index, shifts};
    }

    /* Cell-list algorithm for larger systems */
    double xmin = pos_acc[0][0], xmax = pos_acc[0][0];
    double ymin = pos_acc[0][1], ymax = pos_acc[0][1];
    double zmin = pos_acc[0][2], zmax = pos_acc[0][2];
    for (int64_t i = 1; i < nat; i++) {
        xmin = std::min(xmin, pos_acc[i][0]); xmax = std::max(xmax, pos_acc[i][0]);
        ymin = std::min(ymin, pos_acc[i][1]); ymax = std::max(ymax, pos_acc[i][1]);
        zmin = std::min(zmin, pos_acc[i][2]); zmax = std::max(zmax, pos_acc[i][2]);
    }

    /* Add padding to avoid edge cases */
    xmin -= 0.01; ymin -= 0.01; zmin -= 0.01;
    xmax += 0.01; ymax += 0.01; zmax += 0.01;

    int nx = std::max(1, (int)std::ceil((xmax - xmin) / cutoff));
    int ny = std::max(1, (int)std::ceil((ymax - ymin) / cutoff));
    int nz = std::max(1, (int)std::ceil((zmax - zmin) / cutoff));
    double inv_cutoff = 1.0 / cutoff;

    /* Assign atoms to cells */
    int ncells = nx * ny * nz;
    std::vector<std::vector<int64_t>> cells(ncells);
    for (int64_t i = 0; i < nat; i++) {
        int cx = std::min((int)((pos_acc[i][0] - xmin) * inv_cutoff), nx - 1);
        int cy = std::min((int)((pos_acc[i][1] - ymin) * inv_cutoff), ny - 1);
        int cz = std::min((int)((pos_acc[i][2] - zmin) * inv_cutoff), nz - 1);
        cells[cx * ny * nz + cy * nz + cz].push_back(i);
    }

    /* Build edge list by checking neighboring cells */
    std::vector<int64_t> src_vec, dst_vec;
    src_vec.reserve(nat * 30);
    dst_vec.reserve(nat * 30);

    for (int cx = 0; cx < nx; cx++) {
        for (int cy = 0; cy < ny; cy++) {
            for (int cz = 0; cz < nz; cz++) {
                int cell_idx = cx * ny * nz + cy * nz + cz;
                const auto& cell_atoms = cells[cell_idx];
                if (cell_atoms.empty()) continue;

                /* Check 27 neighboring cells (including self) */
                for (int dx = -1; dx <= 1; dx++) {
                    int nx2 = cx + dx;
                    if (nx2 < 0 || nx2 >= nx) continue;
                    for (int dy = -1; dy <= 1; dy++) {
                        int ny2 = cy + dy;
                        if (ny2 < 0 || ny2 >= ny) continue;
                        for (int dz = -1; dz <= 1; dz++) {
                            int nz2 = cz + dz;
                            if (nz2 < 0 || nz2 >= nz) continue;

                            int nbr_idx = nx2 * ny * nz + ny2 * nz + nz2;
                            const auto& nbr_atoms = cells[nbr_idx];

                            for (int64_t i : cell_atoms) {
                                for (int64_t j : nbr_atoms) {
                                    if (i == j) continue;
                                    double ddx = pos_acc[i][0] - pos_acc[j][0];
                                    double ddy = pos_acc[i][1] - pos_acc[j][1];
                                    double ddz = pos_acc[i][2] - pos_acc[j][2];
                                    if (ddx*ddx + ddy*ddy + ddz*ddz <= cutoff2) {
                                        src_vec.push_back(i);
                                        dst_vec.push_back(j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    int64_t num_edges = static_cast<int64_t>(src_vec.size());
    auto edge_index = torch::empty({2, num_edges}, torch::kInt64);
    std::memcpy(edge_index.data_ptr<int64_t>(),
                src_vec.data(), num_edges * sizeof(int64_t));
    std::memcpy(edge_index.data_ptr<int64_t>() + num_edges,
                dst_vec.data(), num_edges * sizeof(int64_t));
    auto shifts = torch::zeros({num_edges, 3}, torch::kFloat64);

    return {edge_index, shifts};
}

/*
 * Build neighbor list into pre-allocated vectors (for batched use).
 * Positions are a contiguous slice [nat, 3] of a larger tensor.
 */
static void build_neighbor_list_into(
    const double* pos_data, int64_t nat, double cutoff,
    std::vector<int64_t>& src_vec, std::vector<int64_t>& dst_vec)
{
    double cutoff2 = cutoff * cutoff;
    src_vec.clear();
    dst_vec.clear();

    if (nat <= 64) {
        /* Simple O(N²) for small systems */
        for (int64_t i = 0; i < nat; i++) {
            for (int64_t j = 0; j < nat; j++) {
                if (i == j) continue;
                double dx = pos_data[i*3+0] - pos_data[j*3+0];
                double dy = pos_data[i*3+1] - pos_data[j*3+1];
                double dz = pos_data[i*3+2] - pos_data[j*3+2];
                if (dx*dx + dy*dy + dz*dz <= cutoff2) {
                    src_vec.push_back(i);
                    dst_vec.push_back(j);
                }
            }
        }
        return;
    }

    /* Cell-list for larger systems */
    double xmin = pos_data[0], xmax = pos_data[0];
    double ymin = pos_data[1], ymax = pos_data[1];
    double zmin = pos_data[2], zmax = pos_data[2];
    for (int64_t i = 1; i < nat; i++) {
        double x = pos_data[i*3+0], y = pos_data[i*3+1], z = pos_data[i*3+2];
        xmin = std::min(xmin, x); xmax = std::max(xmax, x);
        ymin = std::min(ymin, y); ymax = std::max(ymax, y);
        zmin = std::min(zmin, z); zmax = std::max(zmax, z);
    }
    xmin -= 0.01; ymin -= 0.01; zmin -= 0.01;
    xmax += 0.01; ymax += 0.01; zmax += 0.01;

    int cnx = std::max(1, (int)std::ceil((xmax - xmin) / cutoff));
    int cny = std::max(1, (int)std::ceil((ymax - ymin) / cutoff));
    int cnz = std::max(1, (int)std::ceil((zmax - zmin) / cutoff));
    double inv_cutoff = 1.0 / cutoff;

    int ncells = cnx * cny * cnz;
    std::vector<std::vector<int64_t>> cells(ncells);
    for (int64_t i = 0; i < nat; i++) {
        int cx = std::min((int)((pos_data[i*3+0] - xmin) * inv_cutoff), cnx - 1);
        int cy = std::min((int)((pos_data[i*3+1] - ymin) * inv_cutoff), cny - 1);
        int cz = std::min((int)((pos_data[i*3+2] - zmin) * inv_cutoff), cnz - 1);
        cells[cx * cny * cnz + cy * cnz + cz].push_back(i);
    }

    for (int cx = 0; cx < cnx; cx++) {
        for (int cy = 0; cy < cny; cy++) {
            for (int cz = 0; cz < cnz; cz++) {
                int cell_idx = cx * cny * cnz + cy * cnz + cz;
                const auto& cell_atoms = cells[cell_idx];
                if (cell_atoms.empty()) continue;

                for (int ddx = -1; ddx <= 1; ddx++) {
                    int nx2 = cx + ddx;
                    if (nx2 < 0 || nx2 >= cnx) continue;
                    for (int ddy = -1; ddy <= 1; ddy++) {
                        int ny2 = cy + ddy;
                        if (ny2 < 0 || ny2 >= cny) continue;
                        for (int ddz = -1; ddz <= 1; ddz++) {
                            int nz2 = cz + ddz;
                            if (nz2 < 0 || nz2 >= cnz) continue;

                            int nbr_idx = nx2 * cny * cnz + ny2 * cnz + nz2;
                            const auto& nbr_atoms = cells[nbr_idx];

                            for (int64_t i : cell_atoms) {
                                for (int64_t j : nbr_atoms) {
                                    if (i == j) continue;
                                    double dx = pos_data[i*3+0] - pos_data[j*3+0];
                                    double dy = pos_data[i*3+1] - pos_data[j*3+1];
                                    double dz = pos_data[i*3+2] - pos_data[j*3+2];
                                    if (dx*dx + dy*dy + dz*dz <= cutoff2) {
                                        src_vec.push_back(i);
                                        dst_vec.push_back(j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


/* ====================================================================
 * MACE-specific helpers
 * ==================================================================== */

/*
 * MACE-MH z_table: [1,2,...,83,89,90,91,92,93,94] -> 89 entries
 * Index = Z-1 for Z in 1..83, then index 83..88 for Z in 89..94
 */
static int z_to_mace_index(int z) {
    if (z >= 1 && z <= 83) return z - 1;
    if (z >= 89 && z <= 94) return 83 + (z - 89);
    return -1;  /* unsupported element */
}
static constexpr int MACE_NUM_TYPES = 89;


/* ====================================================================
 * Single-structure MACE inference
 * ==================================================================== */

static int mace_engrad(LibtorchContext* ctx,
                       int nat,
                       const double* positions_bohr,
                       const int* atomic_numbers,
                       double* energy_out,
                       double* gradient_out,
                       char* err_msg, int err_len)
{
    /* Convert positions: Bohr -> Angstrom (compute once) */
    auto pos_bohr = torch::from_blob(
        const_cast<double*>(positions_bohr),
        {nat, 3}, torch::kFloat64);
    auto pos_ang = pos_bohr * BOHR_TO_ANG;

    /* Build neighbor list on CPU in Angstrom space */
    auto [edge_index, shifts] = build_neighbor_list(pos_ang, ctx->cutoff_ang);

    /* Move dynamic tensors to device */
    auto positions = pos_ang.to(ctx->device);
    edge_index = edge_index.to(ctx->device);
    shifts = shifts.to(ctx->device);
    /* Note: requires_grad is set by the MACE model internally in forward() */

    /* Acquire mutex for shared model: protects cached tensors + forward pass.
     * Neighbor list construction above runs in parallel (thread-local data). */
    std::lock_guard<std::mutex> lock(ctx->forward_mutex);

    /* Rebuild cached tensors only when atom count or composition changes */
    bool nat_changed = (nat != ctx->cached_nat);
    bool z_changed = nat_changed;
    if (!nat_changed) {
        for (int i = 0; i < nat && !z_changed; i++) {
            if (ctx->cached_z[i] != atomic_numbers[i]) z_changed = true;
        }
    }

    if (nat_changed) {
        ctx->cached_nat = nat;
        ctx->cached_batch = torch::zeros({nat}, torch::kInt64).to(ctx->device);
        ctx->cached_ptr = torch::tensor({(int64_t)0, (int64_t)nat},
                                         torch::kInt64).to(ctx->device);
        ctx->cached_cell = torch::zeros({3, 3}, torch::kFloat64).to(ctx->device);
        ctx->cached_pbc = torch::zeros({3}, torch::kBool).to(ctx->device);
        ctx->cached_local_or_ghost = torch::ones({nat}, torch::kFloat64).to(ctx->device);
    }

    if (z_changed) {
        ctx->cached_z.assign(atomic_numbers, atomic_numbers + nat);
        auto node_attrs = torch::zeros({nat, MACE_NUM_TYPES}, torch::kFloat64);
        auto na_acc = node_attrs.accessor<double, 2>();
        for (int i = 0; i < nat; i++) {
            int idx = z_to_mace_index(atomic_numbers[i]);
            if (idx < 0) {
                set_error(err_msg, err_len,
                    std::string("Unsupported element Z=") + std::to_string(atomic_numbers[i]));
                return 7;
            }
            na_acc[i][idx] = 1.0;
        }
        ctx->cached_node_attrs = node_attrs.to(ctx->device);
    }

    /* Build the data dict (dynamic: positions, edge_index, shifts; cached: rest) */
    c10::Dict<std::string, torch::Tensor> data;
    data.insert("positions", positions);
    data.insert("node_attrs", ctx->cached_node_attrs);
    data.insert("edge_index", edge_index);
    data.insert("shifts", shifts);
    data.insert("batch", ctx->cached_batch);
    data.insert("ptr", ctx->cached_ptr);
    data.insert("cell", ctx->cached_cell);
    data.insert("pbc", ctx->cached_pbc);

    /* Forward pass */
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(3);
    inputs.push_back(data);
    inputs.push_back(ctx->cached_local_or_ghost);
    inputs.push_back(false);  /* compute_virials = false */

    auto output_ivalue = ctx->module.forward(inputs);
    auto output_dict = output_ivalue.toGenericDict();

    /* Extract energy (eV) */
    auto energy_ivalue = output_dict.at("total_energy_local");
    torch::Tensor energy_tensor;
    if (energy_ivalue.isNone()) {
        set_error(err_msg, err_len, "MACE returned None for energy");
        return 5;
    }
    energy_tensor = energy_ivalue.toTensor().to(torch::kFloat64).to(torch::kCPU).contiguous();
    double energy_ev = energy_tensor.data_ptr<double>()[0];

    /* Extract forces (eV/Angstrom) */
    auto forces_ivalue = output_dict.at("forces");
    if (forces_ivalue.isNone()) {
        set_error(err_msg, err_len, "MACE returned None for forces");
        return 6;
    }
    auto forces_tensor = forces_ivalue.toTensor().to(torch::kFloat64).to(torch::kCPU).contiguous();

    /* Convert: energy eV -> Hartree */
    *energy_out = energy_ev * EV_TO_HARTREE;

    /* Convert: forces eV/Ang -> gradient Hartree/Bohr
     * gradient = -forces, and Hartree/Bohr = eV/Ang * EV_TO_HARTREE / ANG_TO_BOHR
     *          = eV/Ang * EV_TO_HARTREE * BOHR_TO_ANG */
    double conv = EV_TO_HARTREE * BOHR_TO_ANG;  /* == EV_TO_HARTREE / ANG_TO_BOHR */
    auto forces_ptr = forces_tensor.data_ptr<double>();
    for (int i = 0; i < 3 * nat; i++) {
        gradient_out[i] = -forces_ptr[i] * conv;
    }

    return 0;
}


/* ====================================================================
 * Batched MACE inference — multiple structures in one forward pass
 * ==================================================================== */

static int mace_engrad_batch(LibtorchContext* ctx,
                             int batch_size, int nat,
                             const double* positions_bohr,
                             const int* atomic_numbers,
                             double* energies_out,
                             double* gradients_out,
                             char* err_msg, int err_len)
{
    int B = batch_size;
    int64_t total_atoms = (int64_t)B * nat;

    /* 1. Convert all positions: Bohr -> Angstrom */
    auto all_pos_bohr = torch::from_blob(
        const_cast<double*>(positions_bohr),
        {total_atoms, 3}, torch::kFloat64);
    auto all_pos_ang = all_pos_bohr * BOHR_TO_ANG;
    auto pos_ang_ptr = all_pos_ang.data_ptr<double>();

    /* 2. Build neighbor lists in parallel on CPU */
    std::vector<std::vector<int64_t>> all_src(B), all_dst(B);

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; b++) {
        const double* pos_b = pos_ang_ptr + (int64_t)b * nat * 3;
        build_neighbor_list_into(pos_b, nat, ctx->cutoff_ang,
                                 all_src[b], all_dst[b]);
    }

    /* 3. Count total edges and build concatenated edge_index with offsets */
    int64_t total_edges = 0;
    for (int b = 0; b < B; b++) {
        total_edges += (int64_t)all_src[b].size();
    }

    auto edge_index = torch::empty({2, total_edges}, torch::kInt64);
    auto ei_ptr = edge_index.data_ptr<int64_t>();
    int64_t offset = 0;
    for (int b = 0; b < B; b++) {
        int64_t ne = (int64_t)all_src[b].size();
        int64_t atom_offset = (int64_t)b * nat;
        for (int64_t e = 0; e < ne; e++) {
            ei_ptr[offset + e] = all_src[b][e] + atom_offset;
            ei_ptr[total_edges + offset + e] = all_dst[b][e] + atom_offset;
        }
        offset += ne;
    }

    auto shifts = torch::zeros({total_edges, 3}, torch::kFloat64);

    /* 4. Build batch tensor: [0,0,...,0, 1,1,...,1, ..., B-1,...,B-1] */
    auto batch_tensor = torch::empty({total_atoms}, torch::kInt64);
    auto bptr = batch_tensor.data_ptr<int64_t>();
    for (int b = 0; b < B; b++) {
        for (int64_t i = 0; i < nat; i++) {
            bptr[(int64_t)b * nat + i] = b;
        }
    }

    /* 5. Build ptr tensor: [0, nat, 2*nat, ..., B*nat] */
    auto ptr_tensor = torch::empty({(int64_t)B + 1}, torch::kInt64);
    auto pptr = ptr_tensor.data_ptr<int64_t>();
    for (int b = 0; b <= B; b++) {
        pptr[b] = (int64_t)b * nat;
    }

    /* 6. Build node_attrs — tile from single structure */
    auto node_attrs_single = torch::zeros({(int64_t)nat, MACE_NUM_TYPES}, torch::kFloat64);
    {
        auto na_acc = node_attrs_single.accessor<double, 2>();
        for (int i = 0; i < nat; i++) {
            int idx = z_to_mace_index(atomic_numbers[i]);
            if (idx < 0) {
                set_error(err_msg, err_len,
                    std::string("Unsupported element Z=") + std::to_string(atomic_numbers[i]));
                return 7;
            }
            na_acc[i][idx] = 1.0;
        }
    }
    auto node_attrs = node_attrs_single.repeat({B, 1});

    auto local_or_ghost = torch::ones({total_atoms}, torch::kFloat64);
    auto cell = torch::zeros({3, 3}, torch::kFloat64);
    auto pbc = torch::zeros({3}, torch::kBool);

    /* 7. Move everything to device */
    auto positions = all_pos_ang.to(ctx->device);
    edge_index = edge_index.to(ctx->device);
    shifts = shifts.to(ctx->device);
    batch_tensor = batch_tensor.to(ctx->device);
    ptr_tensor = ptr_tensor.to(ctx->device);
    node_attrs = node_attrs.to(ctx->device);
    local_or_ghost = local_or_ghost.to(ctx->device);
    cell = cell.to(ctx->device);
    pbc = pbc.to(ctx->device);

    /* 8. Build data dict */
    c10::Dict<std::string, torch::Tensor> data;
    data.insert("positions", positions);
    data.insert("node_attrs", node_attrs);
    data.insert("edge_index", edge_index);
    data.insert("shifts", shifts);
    data.insert("batch", batch_tensor);
    data.insert("ptr", ptr_tensor);
    data.insert("cell", cell);
    data.insert("pbc", pbc);

    /* 9. Forward pass */
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(3);
    inputs.push_back(data);
    inputs.push_back(local_or_ghost);
    inputs.push_back(false);  /* compute_virials = false */

    auto output_ivalue = ctx->module.forward(inputs);
    auto output_dict = output_ivalue.toGenericDict();

    /* 10. Extract energies [B] in eV */
    auto energy_ivalue = output_dict.at("total_energy_local");
    if (energy_ivalue.isNone()) {
        set_error(err_msg, err_len, "MACE batch returned None for energy");
        return 5;
    }
    auto energy_tensor = energy_ivalue.toTensor().to(torch::kFloat64).to(torch::kCPU).contiguous();
    auto energy_ptr = energy_tensor.data_ptr<double>();

    /* 11. Extract forces [total_atoms, 3] in eV/Angstrom */
    auto forces_ivalue = output_dict.at("forces");
    if (forces_ivalue.isNone()) {
        set_error(err_msg, err_len, "MACE batch returned None for forces");
        return 6;
    }
    auto forces_tensor = forces_ivalue.toTensor().to(torch::kFloat64).to(torch::kCPU).contiguous();
    auto forces_ptr = forces_tensor.data_ptr<double>();

    /* 12. Convert and distribute results */
    double conv = EV_TO_HARTREE * BOHR_TO_ANG;
    for (int b = 0; b < B; b++) {
        energies_out[b] = energy_ptr[b] * EV_TO_HARTREE;
        int64_t base = (int64_t)b * nat * 3;
        for (int64_t i = 0; i < (int64_t)nat * 3; i++) {
            gradients_out[base + i] = -forces_ptr[base + i] * conv;
        }
    }

    return 0;
}


/* ====================================================================
 * Pipelined batch inference — 3-stage functions
 *
 * Stage 1 (prepare): CPU neighbor-list construction + tensor building
 * Stage 2 (launch):  GPU forward pass (async on CUDA)
 * Stage 3 (collect): sync + extract results to output arrays
 *
 * By running prepare(batch K+1) while GPU processes batch K, we overlap
 * CPU and GPU work for higher throughput.
 * ==================================================================== */

static int mace_prepare_batch(
    LibtorchContext* ctx, BatchWorkspace& ws,
    int B, int nat,
    const double* positions_bohr, const int* atomic_numbers,
    char* err_msg, int err_len)
{
    int64_t total_atoms = (int64_t)B * nat;
    ws.bcount = B;
    ws.nat = nat;
    ws.has_output = false;

    /* 1. Convert positions: Bohr -> Angstrom */
    auto all_pos_bohr = torch::from_blob(
        const_cast<double*>(positions_bohr),
        {total_atoms, 3}, torch::kFloat64);
    auto all_pos_ang = all_pos_bohr * BOHR_TO_ANG;
    auto pos_ang_ptr = all_pos_ang.data_ptr<double>();

    /* 2. Build neighbor lists in parallel on CPU */
    std::vector<std::vector<int64_t>> all_src(B), all_dst(B);

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; b++) {
        const double* pos_b = pos_ang_ptr + (int64_t)b * nat * 3;
        build_neighbor_list_into(pos_b, nat, ctx->cutoff_ang,
                                 all_src[b], all_dst[b]);
    }

    /* 3. Concatenate edge_index with per-structure atom offsets */
    int64_t total_edges = 0;
    for (int b = 0; b < B; b++) {
        total_edges += (int64_t)all_src[b].size();
    }

    auto edge_index = torch::empty({2, total_edges}, torch::kInt64);
    auto ei_ptr = edge_index.data_ptr<int64_t>();
    int64_t offset = 0;
    for (int b = 0; b < B; b++) {
        int64_t ne = (int64_t)all_src[b].size();
        int64_t atom_offset = (int64_t)b * nat;
        for (int64_t e = 0; e < ne; e++) {
            ei_ptr[offset + e] = all_src[b][e] + atom_offset;
            ei_ptr[total_edges + offset + e] = all_dst[b][e] + atom_offset;
        }
        offset += ne;
    }

    auto shifts = torch::zeros({total_edges, 3}, torch::kFloat64);

    /* 4. Build batch tensor: [0,0,...,0, 1,1,...,1, ..., B-1,...,B-1] */
    auto batch_tensor = torch::empty({total_atoms}, torch::kInt64);
    auto bptr = batch_tensor.data_ptr<int64_t>();
    for (int b = 0; b < B; b++) {
        for (int64_t i = 0; i < nat; i++) {
            bptr[(int64_t)b * nat + i] = b;
        }
    }

    /* 5. Build ptr tensor: [0, nat, 2*nat, ..., B*nat] */
    auto ptr_tensor = torch::empty({(int64_t)B + 1}, torch::kInt64);
    auto pptr = ptr_tensor.data_ptr<int64_t>();
    for (int b = 0; b <= B; b++) {
        pptr[b] = (int64_t)b * nat;
    }

    /* 6. Build node_attrs — tile from single structure */
    auto node_attrs_single = torch::zeros({(int64_t)nat, MACE_NUM_TYPES},
                                           torch::kFloat64);
    {
        auto na_acc = node_attrs_single.accessor<double, 2>();
        for (int i = 0; i < nat; i++) {
            int idx = z_to_mace_index(atomic_numbers[i]);
            if (idx < 0) {
                set_error(err_msg, err_len,
                    std::string("Unsupported element Z=") +
                    std::to_string(atomic_numbers[i]));
                return 7;
            }
            na_acc[i][idx] = 1.0;
        }
    }
    auto node_attrs = node_attrs_single.repeat({B, 1});

    auto local_or_ghost = torch::ones({total_atoms}, torch::kFloat64);
    auto cell = torch::zeros({3, 3}, torch::kFloat64);
    auto pbc_tensor = torch::zeros({3}, torch::kBool);

    /* 7. Move tensors to device */
    ws.positions     = all_pos_ang.to(ctx->device);
    ws.edge_index    = edge_index.to(ctx->device);
    ws.shifts        = shifts.to(ctx->device);
    ws.batch_tensor  = batch_tensor.to(ctx->device);
    ws.ptr_tensor    = ptr_tensor.to(ctx->device);
    ws.node_attrs    = node_attrs.to(ctx->device);
    ws.local_or_ghost = local_or_ghost.to(ctx->device);
    ws.cell          = cell.to(ctx->device);
    ws.pbc           = pbc_tensor.to(ctx->device);

    return 0;
}


static int mace_launch_batch(LibtorchContext* ctx, BatchWorkspace& ws)
{
    /* Build data dict from workspace tensors */
    c10::Dict<std::string, torch::Tensor> data;
    data.insert("positions",  ws.positions);
    data.insert("node_attrs", ws.node_attrs);
    data.insert("edge_index", ws.edge_index);
    data.insert("shifts",     ws.shifts);
    data.insert("batch",      ws.batch_tensor);
    data.insert("ptr",        ws.ptr_tensor);
    data.insert("cell",       ws.cell);
    data.insert("pbc",        ws.pbc);

    /* Forward pass — async on CUDA (returns immediately, GPU kernels queued) */
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(3);
    inputs.push_back(data);
    inputs.push_back(ws.local_or_ghost);
    inputs.push_back(false);  /* compute_virials = false */

    ws.output = ctx->module.forward(inputs);
    ws.has_output = true;

    return 0;
}


static int mace_collect_batch(
    BatchWorkspace& ws,
    double* energies_out, double* gradients_out,
    char* err_msg, int err_len)
{
    if (!ws.has_output) {
        set_error(err_msg, err_len, "mace_collect_batch: no output to collect");
        return 9;
    }

    int B = ws.bcount;
    int nat = ws.nat;

    auto output_dict = ws.output.toGenericDict();

    /* Extract energies [B] in eV — accessing GPU tensor forces sync */
    auto energy_ivalue = output_dict.at("total_energy_local");
    if (energy_ivalue.isNone()) {
        set_error(err_msg, err_len, "MACE batch returned None for energy");
        return 5;
    }
    auto energy_tensor = energy_ivalue.toTensor()
        .to(torch::kFloat64).to(torch::kCPU).contiguous();
    auto energy_ptr = energy_tensor.data_ptr<double>();

    /* Extract forces [total_atoms, 3] in eV/Angstrom */
    auto forces_ivalue = output_dict.at("forces");
    if (forces_ivalue.isNone()) {
        set_error(err_msg, err_len, "MACE batch returned None for forces");
        return 6;
    }
    auto forces_tensor = forces_ivalue.toTensor()
        .to(torch::kFloat64).to(torch::kCPU).contiguous();
    auto forces_ptr = forces_tensor.data_ptr<double>();

    /* Convert and distribute results */
    double conv = EV_TO_HARTREE * BOHR_TO_ANG;
    for (int b = 0; b < B; b++) {
        energies_out[b] = energy_ptr[b] * EV_TO_HARTREE;
        int64_t base = (int64_t)b * nat * 3;
        for (int64_t i = 0; i < (int64_t)nat * 3; i++) {
            gradients_out[base + i] = -forces_ptr[base + i] * conv;
        }
    }

    /* Release output to free GPU memory */
    ws.output = torch::jit::IValue();
    ws.has_output = false;

    return 0;
}


/* Unified pipelined batch inference for 1..N GPUs.
 * Double-buffers (or N-buffers) to overlap CPU NL construction with GPU forward.
 *
 * contexts[ngpus] — one LibtorchContext per GPU (all loaded with same model)
 * Batches are round-robin across GPUs with workspace recycling. */
static int mace_pipeline_dispatch(
    LibtorchContext** contexts, int ngpus,
    int total_structures, int nat,
    const double* all_positions_bohr, const int* atomic_numbers,
    int batch_size,
    double* energies_out, double* gradients_out,
    char* err_msg, int err_len)
{
    int nbatches = (total_structures + batch_size - 1) / batch_size;
    if (nbatches == 0) return 0;

    /* Need at least 2 workspaces for pipelining (even with 1 GPU) */
    int num_ws = std::max(ngpus, 2);
    std::vector<BatchWorkspace> ws(num_ws);

    int rc = 0;

    for (int ib = 0; ib < nbatches; ib++) {
        int ws_cur = ib % num_ws;
        int gpu_cur = ib % ngpus;
        auto ctx_cur = contexts[gpu_cur];

        /* Compute batch range */
        int bstart = ib * batch_size;
        int bcount = std::min(batch_size, total_structures - bstart);

        /* Stage 1: Prepare new batch (CPU: NL construction + tensor build).
         * This overlaps with GPU processing the previous batch. */
        rc = mace_prepare_batch(ctx_cur, ws[ws_cur], bcount, nat,
                                all_positions_bohr + (int64_t)bstart * nat * 3,
                                atomic_numbers, err_msg, err_len);
        if (rc != 0) return rc;

        /* Stage 3 (of previous batch): Collect results.
         * GPU should be done by now (NL construction took time). */
        if (ib > 0) {
            int ws_prev = (ib - 1) % num_ws;
            int prev_start = (ib - 1) * batch_size;
            int prev_count = std::min(batch_size, total_structures - prev_start);
            rc = mace_collect_batch(ws[ws_prev],
                                    energies_out + prev_start,
                                    gradients_out + (int64_t)prev_start * nat * 3,
                                    err_msg, err_len);
            if (rc != 0) return rc;
        }

        /* Stage 2: Launch forward on GPU (async on CUDA) */
        rc = mace_launch_batch(ctx_cur, ws[ws_cur]);
        if (rc != 0) return rc;
    }

    /* Collect last batch */
    {
        int last_ws = (nbatches - 1) % num_ws;
        int last_start = (nbatches - 1) * batch_size;
        rc = mace_collect_batch(ws[last_ws],
                                energies_out + last_start,
                                gradients_out + (int64_t)last_start * nat * 3,
                                err_msg, err_len);
    }

    return rc;
}


/* ====================================================================
 * C API implementation
 * ==================================================================== */

extern "C" {

void libtorch_set_num_threads(int aten_threads)
{
    if (aten_threads > 0) {
        at::set_num_threads(aten_threads);
    }
    at::set_num_interop_threads(1);
}


libtorch_handle_t libtorch_load(const char* model_path, int device_code,
                                 int model_format, double cutoff,
                                 char* err_msg, int err_len)
{
    try {
        auto ctx = new LibtorchContext();
        ctx->device = device_from_int(device_code);
        ctx->model_format = model_format;
        ctx->cutoff_ang = cutoff > 0 ? cutoff : 6.0;

        /* Load TorchScript module */
        ctx->module = torch::jit::load(model_path, ctx->device);
        ctx->module.eval();

        /* Optimize for inference */
        torch::jit::setGraphExecutorOptimize(true);

        ctx->initialized = true;
        return static_cast<libtorch_handle_t>(ctx);

    } catch (const c10::Error& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch load error: ") + e.what());
        return nullptr;
    } catch (const std::exception& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch load error: ") + e.what());
        return nullptr;
    }
}


libtorch_handle_t libtorch_load_shared(const char* model_path, int device_code,
                                        int model_format, double cutoff,
                                        char* err_msg, int err_len)
{
    auto ctx = g_shared_models.get_or_load(model_path, device_code,
                                            model_format, cutoff,
                                            err_msg, err_len);
    return static_cast<libtorch_handle_t>(ctx);
}


int libtorch_engrad(libtorch_handle_t handle,
                    int nat,
                    const double* positions_bohr,
                    const int* atomic_numbers,
                    double* energy_out,
                    double* gradient_out,
                    char* err_msg, int err_len)
{
    if (!handle) {
        set_error(err_msg, err_len, "libtorch_engrad: null handle");
        return 1;
    }

    auto ctx = static_cast<LibtorchContext*>(handle);
    if (!ctx->initialized) {
        set_error(err_msg, err_len, "libtorch_engrad: model not initialized");
        return 2;
    }

    if (nat <= 0) {
        set_error(err_msg, err_len, "libtorch_engrad: invalid nat (must be > 0)");
        if (energy_out) *energy_out = 0.0;
        return 10;
    }

    /* Defensively zero outputs so error/exception paths never return garbage */
    *energy_out = 0.0;
    std::memset(gradient_out, 0, sizeof(double) * 3 * nat);

    try {
        if (ctx->model_format == LIBTORCH_FMT_MACE) {
            return mace_engrad(ctx, nat, positions_bohr, atomic_numbers,
                               energy_out, gradient_out, err_msg, err_len);
        }

        /* Generic format: forward(positions, Z) -> (energy, gradient) */
        torch::InferenceMode infer_guard;

        auto pos_tensor = torch::from_blob(
            const_cast<double*>(positions_bohr),
            {nat, 3},
            torch::TensorOptions().dtype(torch::kFloat64)
        ).to(ctx->device);

        auto z_tensor = torch::from_blob(
            const_cast<int*>(atomic_numbers),
            {nat},
            torch::TensorOptions().dtype(torch::kInt32)
        ).to(torch::kInt64).to(ctx->device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pos_tensor);
        inputs.push_back(z_tensor);

        auto output = ctx->module.forward(inputs).toTuple();

        auto energy_tensor = output->elements()[0].toTensor()
            .to(torch::kFloat64).to(torch::kCPU).contiguous();
        auto gradient_tensor = output->elements()[1].toTensor()
            .to(torch::kFloat64).to(torch::kCPU).contiguous();

        *energy_out = energy_tensor.data_ptr<double>()[0];
        std::memcpy(gradient_out, gradient_tensor.data_ptr<double>(),
                     sizeof(double) * 3 * nat);

        return 0;

    } catch (const c10::Error& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch inference error: ") + e.what());
        return 3;
    } catch (const std::exception& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch inference error: ") + e.what());
        return 4;
    }
}


int libtorch_engrad_batch(libtorch_handle_t handle,
                          int batch_size, int nat,
                          const double* positions_bohr,
                          const int* atomic_numbers,
                          double* energies_out,
                          double* gradients_out,
                          char* err_msg, int err_len)
{
    if (!handle) {
        set_error(err_msg, err_len, "libtorch_engrad_batch: null handle");
        return 1;
    }

    auto ctx = static_cast<LibtorchContext*>(handle);
    if (!ctx->initialized) {
        set_error(err_msg, err_len, "libtorch_engrad_batch: model not initialized");
        return 2;
    }

    if (batch_size <= 0 || nat <= 0) {
        set_error(err_msg, err_len, "libtorch_engrad_batch: invalid batch_size or nat");
        return 8;
    }

    try {
        if (ctx->model_format == LIBTORCH_FMT_MACE) {
            /* Lock mutex for shared model access during forward pass */
            std::lock_guard<std::mutex> lock(ctx->forward_mutex);
            return mace_engrad_batch(ctx, batch_size, nat, positions_bohr,
                                     atomic_numbers, energies_out, gradients_out,
                                     err_msg, err_len);
        }

        /* Generic format: fall back to sequential single-structure calls */
        for (int b = 0; b < batch_size; b++) {
            int rc = libtorch_engrad(handle, nat,
                                     positions_bohr + (int64_t)b * nat * 3,
                                     atomic_numbers,
                                     &energies_out[b],
                                     gradients_out + (int64_t)b * nat * 3,
                                     err_msg, err_len);
            if (rc != 0) return rc;
        }
        return 0;

    } catch (const c10::Error& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch batch inference error: ") + e.what());
        return 3;
    } catch (const std::exception& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch batch inference error: ") + e.what());
        return 4;
    }
}


int libtorch_engrad_batch_pipeline(
    libtorch_handle_t handle,
    int total_structures, int nat,
    const double* all_positions_bohr,
    const int* atomic_numbers,
    int batch_size,
    double* energies_out,
    double* gradients_out,
    char* err_msg, int err_len)
{
    if (!handle) {
        set_error(err_msg, err_len, "libtorch_engrad_batch_pipeline: null handle");
        return 1;
    }
    auto ctx = static_cast<LibtorchContext*>(handle);
    if (!ctx->initialized) {
        set_error(err_msg, err_len, "libtorch_engrad_batch_pipeline: not initialized");
        return 2;
    }
    if (total_structures <= 0 || nat <= 0 || batch_size <= 0) {
        set_error(err_msg, err_len, "libtorch_engrad_batch_pipeline: invalid parameters");
        return 8;
    }

    try {
        if (ctx->model_format != LIBTORCH_FMT_MACE) {
            /* Non-MACE: fall back to sequential batch calls */
            for (int b = 0; b < total_structures; b++) {
                int rc = libtorch_engrad(handle, nat,
                    all_positions_bohr + (int64_t)b * nat * 3,
                    atomic_numbers, &energies_out[b],
                    gradients_out + (int64_t)b * nat * 3,
                    err_msg, err_len);
                if (rc != 0) return rc;
            }
            return 0;
        }

        LibtorchContext* ctxs[1] = { ctx };
        return mace_pipeline_dispatch(ctxs, 1,
            total_structures, nat, all_positions_bohr, atomic_numbers,
            batch_size, energies_out, gradients_out, err_msg, err_len);

    } catch (const c10::Error& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch pipeline error: ") + e.what());
        return 3;
    } catch (const std::exception& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch pipeline error: ") + e.what());
        return 4;
    }
}


int libtorch_engrad_batch_multigpu(
    libtorch_handle_t* handles, int ngpus,
    int total_structures, int nat,
    const double* all_positions_bohr,
    const int* atomic_numbers,
    int batch_size,
    double* energies_out,
    double* gradients_out,
    char* err_msg, int err_len)
{
    if (!handles || ngpus <= 0) {
        set_error(err_msg, err_len, "libtorch_engrad_batch_multigpu: invalid handles/ngpus");
        return 1;
    }
    if (total_structures <= 0 || nat <= 0 || batch_size <= 0) {
        set_error(err_msg, err_len, "libtorch_engrad_batch_multigpu: invalid parameters");
        return 8;
    }

    /* Validate all handles */
    std::vector<LibtorchContext*> ctxs(ngpus);
    for (int g = 0; g < ngpus; g++) {
        if (!handles[g]) {
            set_error(err_msg, err_len, "libtorch_engrad_batch_multigpu: null handle");
            return 1;
        }
        ctxs[g] = static_cast<LibtorchContext*>(handles[g]);
        if (!ctxs[g]->initialized) {
            set_error(err_msg, err_len, "libtorch_engrad_batch_multigpu: not initialized");
            return 2;
        }
        if (ctxs[g]->model_format != LIBTORCH_FMT_MACE) {
            set_error(err_msg, err_len,
                "libtorch_engrad_batch_multigpu: only MACE format supported");
            return 10;
        }
    }

    try {
        return mace_pipeline_dispatch(ctxs.data(), ngpus,
            total_structures, nat, all_positions_bohr, atomic_numbers,
            batch_size, energies_out, gradients_out, err_msg, err_len);

    } catch (const c10::Error& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch multigpu error: ") + e.what());
        return 3;
    } catch (const std::exception& e) {
        set_error(err_msg, err_len,
                  std::string("libtorch multigpu error: ") + e.what());
        return 4;
    }
}


libtorch_handle_t libtorch_load_shared_on_device(
    const char* model_path, int cuda_device_index,
    int model_format, double cutoff,
    char* err_msg, int err_len)
{
    /* Encode as 10+idx for the device_from_int mapping */
    int device_code = 10 + cuda_device_index;
    return libtorch_load_shared(model_path, device_code,
                                 model_format, cutoff, err_msg, err_len);
}


int libtorch_get_cuda_device_count(void)
{
    try {
        if (!torch::cuda::is_available()) return 0;
        return (int)torch::cuda::device_count();
    } catch (...) {
        return 0;
    }
}


void libtorch_free(libtorch_handle_t handle)
{
    if (handle) {
        auto ctx = static_cast<LibtorchContext*>(handle);
        delete ctx;
    }
}


void libtorch_shared_free_all(void)
{
    g_shared_models.free_all();
}


int libtorch_has_cuda(void)
{
    return torch::cuda::is_available() ? 1 : 0;
}


int libtorch_has_mps(void)
{
#ifdef __APPLE__
    try {
        return torch::mps::is_available() ? 1 : 0;
    } catch (...) {
        return 0;
    }
#else
    return 0;
#endif
}


int libtorch_get_gpu_memory(long long* total_bytes, long long* free_bytes)
{
    *total_bytes = 0;
    *free_bytes = 0;
    if (!torch::cuda::is_available()) return 1;
    try {
        size_t free_mem = 0, total_mem = 0;
        /* Use PyTorch's CUDA memory info (wraps cudaMemGetInfo) */
        auto device = torch::Device(torch::kCUDA, 0);
        c10::cuda::CUDAGuard guard(device);
        cudaMemGetInfo(&free_mem, &total_mem);
        *total_bytes = (long long)total_mem;
        *free_bytes = (long long)free_mem;
        return 0;
    } catch (...) {
        return 1;
    }
}

} /* extern "C" */

#else /* !WITH_LIBTORCH */

#include <cstring>

static void set_error_stub(char* err_msg, int err_len, const char* msg) {
    if (err_msg && err_len > 0) {
        int n = (int)strlen(msg);
        if (n >= err_len) n = err_len - 1;
        memcpy(err_msg, msg, n);
        err_msg[n] = '\0';
    }
}

extern "C" {

void libtorch_set_num_threads(int) {}

libtorch_handle_t libtorch_load(const char*, int, int, double,
                                 char* err_msg, int err_len) {
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return nullptr;
}

libtorch_handle_t libtorch_load_shared(const char*, int, int, double,
                                        char* err_msg, int err_len) {
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return nullptr;
}

int libtorch_engrad(libtorch_handle_t, int nat, const double*, const int*,
                    double* energy_out, double* gradient_out,
                    char* err_msg, int err_len) {
    if (energy_out) *energy_out = 0.0;
    if (gradient_out && nat > 0) std::memset(gradient_out, 0, sizeof(double) * 3 * nat);
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return 1;
}

int libtorch_engrad_batch(libtorch_handle_t, int, int, const double*, const int*,
                          double*, double*, char* err_msg, int err_len) {
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return 1;
}

int libtorch_engrad_batch_pipeline(libtorch_handle_t, int, int,
                                    const double*, const int*, int,
                                    double*, double*,
                                    char* err_msg, int err_len) {
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return 1;
}

int libtorch_engrad_batch_multigpu(libtorch_handle_t*, int, int, int,
                                    const double*, const int*, int,
                                    double*, double*,
                                    char* err_msg, int err_len) {
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return 1;
}

libtorch_handle_t libtorch_load_shared_on_device(const char*, int, int, double,
                                                   char* err_msg, int err_len) {
    set_error_stub(err_msg, err_len, "libtorch not compiled (need -DWITH_LIBTORCH)");
    return nullptr;
}

int libtorch_get_cuda_device_count(void) { return 0; }

void libtorch_free(libtorch_handle_t) {}
void libtorch_shared_free_all(void) {}
int libtorch_has_cuda(void) { return 0; }
int libtorch_has_mps(void) { return 0; }
int libtorch_get_gpu_memory(long long* total_bytes, long long* free_bytes) {
    *total_bytes = 0; *free_bytes = 0; return 1;
}

} /* extern "C" */

#endif /* WITH_LIBTORCH */
