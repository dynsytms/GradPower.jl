# Device clusters and cluster-contiguous state layout.
#
# A device cluster groups a generator with all controllers attached to it
# (governor, exciter, stabilizer). The cluster, not the individual device,
# is the unit of block-diagonal structure in the Jacobian (see proposal
# section 3.2). ZIPLoads are not clusters — they live entirely in Y.
#
# This file is included from src/GradPower.jl AFTER coupling.jl (which
# defines _normalize_id and the attaches_to traits) and AFTER all device
# structs are in scope.

# -----------------------------------------------------------------------
# DeviceCluster
# -----------------------------------------------------------------------

struct DeviceCluster
    gen_idx::Int          # index into psd.devices for the generator
    gov_idx::Int          # governor device index (0 = none)
    exc_idx::Int          # exciter device index (0 = none)
    pss_idx::Int          # stabilizer device index (0 = none)
    bus::Int              # internal bus index
    d_k::Int              # total diff states in cluster
    a_k::Int              # total alg states in cluster
    w_size::Int           # |w_k| = d_k + a_k
    w_start::Int          # first z-index of this cluster's states (1-based)
    w_end::Int            # last z-index of this cluster's states
    trivial::Bool         # true for StaticGenerator clusters (|w_k| <= 2)
end

# Type tuple for grouping: canonical tuple of device type symbols present
# in the cluster.
function cluster_type_tuple(cl::DeviceCluster, devices::Vector{DynamicDevice})
    syms = Symbol[_device_symbol(devices[cl.gen_idx].dtype)]
    cl.gov_idx > 0 && push!(syms, _device_symbol(devices[cl.gov_idx].dtype))
    cl.exc_idx > 0 && push!(syms, _device_symbol(devices[cl.exc_idx].dtype))
    cl.pss_idx > 0 && push!(syms, _device_symbol(devices[cl.pss_idx].dtype))
    return Tuple(syms)
end

_device_symbol(::Genrou)           = :genrou
_device_symbol(::IEESGO)           = :ieesgo
_device_symbol(::TGOV1)            = :tgov1
_device_symbol(::SEXS)             = :sexs
_device_symbol(::ESDC1A)           = :esdc1a
_device_symbol(::IEEEST)           = :ieeest
_device_symbol(::StaticGenerator)  = :static_gen
_device_symbol(::ZIPLoad)          = :zipload

# -----------------------------------------------------------------------
# ClusterTable
# -----------------------------------------------------------------------

struct ClusterTable
    clusters::Vector{DeviceCluster}
    # type_groups[i] = (type_tuple, first_cluster_idx, last_cluster_idx)
    type_groups::Vector{Tuple{NTuple{N,Symbol} where N, Int, Int}}
end

# -----------------------------------------------------------------------
# build_clusters!
# -----------------------------------------------------------------------

"""
    build_clusters!(psd, dmap) -> ClusterTable

Walk `psd.devices` and group generators with their attached controllers.
Returns a ClusterTable with clusters sorted by type-tuple (so same-shape
clusters are contiguous). Does NOT yet assign z-indices (w_start/w_end
are set to 0); call `reorder_state!` to compute and apply the layout.

`dmap` is used for `dmap.bus[i]` to get the internal bus index.
"""
function build_clusters!(psd, dmap)
    devices = psd.devices
    ndev = length(devices)

    # Step 1: identify generators (Genrou and StaticGenerator)
    gen_indices = Int[]
    for (i, dev) in enumerate(devices)
        if dev.dtype isa AbstractGeneratorType || dev.dtype isa StaticGenerator
            push!(gen_indices, i)
        end
    end

    # Step 2: for each controller, find which generator it attaches to.
    # Build gen_idx -> (gov_idx, exc_idx, pss_idx) mapping.
    gov_map = Dict{Int,Int}()   # gen device index -> governor device index
    exc_map = Dict{Int,Int}()   # gen device index -> exciter device index
    pss_map = Dict{Int,Int}()   # gen device index -> PSS device index

    for (i, dev) in enumerate(devices)
        target_class = attaches_to(typeof(dev.dtype))
        target_class === nothing && continue
        dev.dtype isa AbstractLoadType && continue

        # Find target by (bus, id) — same logic as wire_controls!
        bus_match = dev.dtype.bus
        id_match  = _normalize_id(dev.dtype.id)

        if dev.dtype isa AbstractStabilizerType
            # PSS attaches to exciter; find exciter, then find its generator
            exc_idx = 0
            for (j, cand) in enumerate(devices)
                if cand.dtype isa target_class &&
                   cand.dtype.bus == bus_match &&
                   _normalize_id(cand.dtype.id) == id_match
                    exc_idx = j
                    break
                end
            end
            if exc_idx > 0
                # Find the generator the exciter attaches to
                exc_target = attaches_to(typeof(devices[exc_idx].dtype))
                if exc_target !== nothing
                    exc_bus = devices[exc_idx].dtype.bus
                    exc_id  = _normalize_id(devices[exc_idx].dtype.id)
                    for (g, gdev) in enumerate(devices)
                        if gdev.dtype isa exc_target &&
                           gdev.dtype.bus == exc_bus &&
                           _normalize_id(gdev.dtype.id) == exc_id
                            pss_map[g] = i
                            break
                        end
                    end
                end
            end
        elseif dev.dtype isa AbstractGovernorType
            for (j, cand) in enumerate(devices)
                if cand.dtype isa target_class &&
                   cand.dtype.bus == bus_match &&
                   _normalize_id(cand.dtype.id) == id_match
                    gov_map[j] = i
                    break
                end
            end
        elseif dev.dtype isa AbstractExciterType
            for (j, cand) in enumerate(devices)
                if cand.dtype isa target_class &&
                   cand.dtype.bus == bus_match &&
                   _normalize_id(cand.dtype.id) == id_match
                    exc_map[j] = i
                    break
                end
            end
        end
    end

    # Step 3: build cluster list (unordered, z-indices not yet assigned)
    raw_clusters = DeviceCluster[]
    for gi in gen_indices
        gen = devices[gi]
        gov = get(gov_map, gi, 0)
        exc = get(exc_map, gi, 0)
        pss = get(pss_map, gi, 0)
        bus = dmap.bus[gi]

        # Count diff and alg states across all devices in cluster
        d_k = gen.dtype.diff_size
        a_k = gen.dtype.alg_size
        if gov > 0
            d_k += devices[gov].dtype.diff_size
            a_k += devices[gov].dtype.alg_size
        end
        if exc > 0
            d_k += devices[exc].dtype.diff_size
            a_k += devices[exc].dtype.alg_size
        end
        if pss > 0
            d_k += devices[pss].dtype.diff_size
            a_k += devices[pss].dtype.alg_size
        end
        w_size = d_k + a_k
        trivial = gen.dtype isa StaticGenerator

        push!(raw_clusters, DeviceCluster(gi, gov, exc, pss, bus,
                                           d_k, a_k, w_size, 0, 0, trivial))
    end

    # Step 4: sort by type-tuple so same-shape clusters are contiguous
    sort!(raw_clusters, by = cl -> cluster_type_tuple(cl, devices))

    # Step 5: identify type-tuple groups
    type_groups = Tuple{NTuple{N,Symbol} where N, Int, Int}[]
    if !isempty(raw_clusters)
        cur_tt = cluster_type_tuple(raw_clusters[1], devices)
        group_start = 1
        for i in 2:length(raw_clusters)
            tt = cluster_type_tuple(raw_clusters[i], devices)
            if tt != cur_tt
                push!(type_groups, (cur_tt, group_start, i - 1))
                cur_tt = tt
                group_start = i
            end
        end
        push!(type_groups, (cur_tt, group_start, length(raw_clusters)))
    end

    return ClusterTable(raw_clusters, type_groups)
end

# -----------------------------------------------------------------------
# reorder_state! — compute cluster-contiguous z-indices and update all
# pointers on DynamicDevice and SoA tables.
# -----------------------------------------------------------------------

"""
    reorder_state!(psd, cluster_table) -> perm

Assign cluster-contiguous z-indices to the ClusterTable entries,
update all diff_ptr/alg_ptr on DynamicDevice and SoA table columns,
and update uvec_idx. Also builds `psd.diff_indices` — a sorted list
of z-positions that are differential states (needed by beuler/Jacobian
scaling which must know which rows get the backward-Euler treatment).

Returns the permutation vector mapping old->new z-indices
(perm[old] = new), useful for permuting an existing z vector.

After this call, z = [w_1 | w_2 | ... | w_Nc | v] where each w_k
contains the cluster's diff AND alg states contiguously, in the order:
generator diff, generator alg, governor diff, governor alg,
exciter diff, exciter alg, stabilizer diff, stabilizer alg.

The v-net block stays at the end (positions diff_dim+alg_dim+1 onward).
"""
function reorder_state!(psd, cluster_table::ClusterTable)
    devices = psd.devices
    diff_dim = psd.diff_dim
    alg_dim  = psd.alg_dim
    n_da = diff_dim + alg_dim  # total device states

    # Build old-position -> new-position permutation.
    # For each cluster, assign contiguous z-positions.
    # old_to_new[old_z_idx] = new_z_idx
    old_to_new = zeros(Int, n_da)

    # Track which NEW positions are differential states
    diff_indices = Int[]

    pos = 1  # next free position in the reordered z
    new_clusters = DeviceCluster[]

    for cl in cluster_table.clusters
        w_start = pos
        # For each device in the cluster (gen, gov, exc, pss), map its
        # old diff and alg positions to the new contiguous block.
        for dev_idx in _cluster_device_order(cl)
            dev = devices[dev_idx]
            ds = dev.dtype.diff_size
            as = dev.dtype.alg_size

            # Map diff states
            for j in 0:(ds-1)
                old_pos = dev.diff_ptr + j
                old_to_new[old_pos] = pos
                push!(diff_indices, pos)
                pos += 1
            end
            # Map alg states
            for j in 0:(as-1)
                old_pos = diff_dim + dev.alg_ptr + j
                old_to_new[old_pos] = pos
                pos += 1
            end
        end
        w_end = pos - 1
        @assert w_end - w_start + 1 == cl.w_size "Cluster size mismatch: expected $(cl.w_size), got $(w_end - w_start + 1)"
        push!(new_clusters, DeviceCluster(cl.gen_idx, cl.gov_idx, cl.exc_idx, cl.pss_idx,
                                           cl.bus, cl.d_k, cl.a_k, cl.w_size,
                                           w_start, w_end, cl.trivial))
    end

    @assert pos - 1 == n_da "Not all device states assigned: pos-1=$(pos-1), n_da=$n_da"
    @assert length(diff_indices) == diff_dim "diff_indices count $(length(diff_indices)) != diff_dim $diff_dim"

    sort!(diff_indices)
    psd.diff_indices = diff_indices

    # Precompute is_diff mask for allocation-free Jacobian scaling
    is_diff = falses(n_da)
    for i in diff_indices
        is_diff[i] = true
    end
    psd.is_diff = is_diff

    # Update cluster_table.clusters in place
    resize!(cluster_table.clusters, length(new_clusters))
    cluster_table.clusters .= new_clusters

    # Network voltages are identity-mapped (they stay at the end)
    # Their old positions are n_da+1 : n_da+2*nbus, new positions same.

    # Update DynamicDevice pointers
    for dev in devices
        ds = dev.dtype.diff_size
        as = dev.dtype.alg_size
        if ds > 0
            dev.diff_ptr = old_to_new[dev.diff_ptr]
        end
        if as > 0
            dev.alg_ptr = old_to_new[diff_dim + dev.alg_ptr] - diff_dim
        end
    end

    # Update uvec_idx: each entry is a z-index pointing at a device state
    if psd.uvec_idx !== nothing
        for i in eachindex(psd.uvec_idx)
            old_idx = psd.uvec_idx[i]
            if old_idx > 0 && old_idx <= n_da
                psd.uvec_idx[i] = old_to_new[old_idx]
            end
            # entries pointing at network voltages (> n_da) stay unchanged
        end
    end

    # Update SoA table pointer columns
    _update_table_pointers!(psd.layout, old_to_new, diff_dim)

    return old_to_new
end

"""
    _cluster_device_order(cl::DeviceCluster) -> Vector{Int}

Return the device indices in the canonical order within a cluster:
generator, governor, exciter, stabilizer. Skips absent devices (index 0).
"""
function _cluster_device_order(cl::DeviceCluster)
    order = Int[]
    push!(order, cl.gen_idx)
    cl.gov_idx > 0 && push!(order, cl.gov_idx)
    cl.exc_idx > 0 && push!(order, cl.exc_idx)
    cl.pss_idx > 0 && push!(order, cl.pss_idx)
    return order
end

"""
    _update_table_pointers!(layout, old_to_new, diff_dim)

Update diff_ptr, alg_ptr, w_idx, efd_idx, pm_idx, vr_idx, vs_idx on all
SoA tables to reflect the reordered state vector. Each table type has
its own pointer semantics (some store absolute z-indices, some store
offsets relative to diff_dim).
"""
function _update_table_pointers!(layout::SimulationLayout, old_to_new::Vector{Int}, diff_dim::Int)
    n_da = length(old_to_new)

    # Genrou: diff_ptr is absolute, alg_ptr is offset (kernel does diff_dim + alg_ptr)
    _remap_genrou!(layout.genrou, old_to_new, diff_dim, n_da)
    # IEESGO: diff_ptr absolute, alg_ptr offset (kernel does alg_ptr + diff_dim)
    _remap_ieesgo!(layout.ieesgo, old_to_new, diff_dim, n_da)
    # TGOV1: same as IEESGO
    _remap_tgov1!(layout.tgov1, old_to_new, diff_dim, n_da)
    # SEXS: diff_ptr absolute, no alg_ptr
    _remap_sexs!(layout.sexs, old_to_new, diff_dim, n_da)
    # ESDC1A: diff_ptr absolute, no alg_ptr
    _remap_esdc1a!(layout.esdc1a, old_to_new, diff_dim, n_da)
    # IEEEST: diff_ptr absolute, alg_ptr offset (kernel does diff_dim + alg_ptr)
    _remap_ieeest!(layout.ieeest, old_to_new, diff_dim, n_da)
    # ZIPLoad: no diff/alg states
    # StaticGenerator: alg_ptr is absolute
    _remap_static_gen!(layout.static_gen, old_to_new, diff_dim, n_da)
    return nothing
end

function _remap_genrou!(table::GenrouTable, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_dp = Int(table.diff_ptr[k])
        table.diff_ptr[k] = Int32(old_to_new[old_dp])

        old_ap_global = Int(table.alg_ptr[k]) + diff_dim
        table.alg_ptr[k] = Int32(old_to_new[old_ap_global] - diff_dim)

        # pm_idx: z-index of governor's p_m alg state (0 if no governor)
        old_pm = Int(table.pm_idx[k])
        if old_pm > 0 && old_pm <= n_da
            table.pm_idx[k] = Int32(old_to_new[old_pm])
        end

        # efd_idx: z-index of exciter's e_fd diff state (0 if no exciter)
        old_efd = Int(table.efd_idx[k])
        if old_efd > 0 && old_efd <= n_da
            table.efd_idx[k] = Int32(old_to_new[old_efd])
        end
    end
    return nothing
end

function _remap_ieesgo!(table::IEESGOTable, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_dp = Int(table.diff_ptr[k])
        table.diff_ptr[k] = Int32(old_to_new[old_dp])

        old_ap_global = Int(table.alg_ptr[k]) + diff_dim
        table.alg_ptr[k] = Int32(old_to_new[old_ap_global] - diff_dim)

        # w_idx: z-index of generator's omega state
        old_w = Int(table.w_idx[k])
        if old_w > 0 && old_w <= n_da
            table.w_idx[k] = Int32(old_to_new[old_w])
        end
    end
    return nothing
end

function _remap_tgov1!(table::TGOV1Table, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_dp = Int(table.diff_ptr[k])
        table.diff_ptr[k] = Int32(old_to_new[old_dp])

        old_ap_global = Int(table.alg_ptr[k]) + diff_dim
        table.alg_ptr[k] = Int32(old_to_new[old_ap_global] - diff_dim)

        old_w = Int(table.w_idx[k])
        if old_w > 0 && old_w <= n_da
            table.w_idx[k] = Int32(old_to_new[old_w])
        end
    end
    return nothing
end

function _remap_sexs!(table::SEXSTable, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_dp = Int(table.diff_ptr[k])
        table.diff_ptr[k] = Int32(old_to_new[old_dp])

        # vr_idx is a network voltage index (> n_da), stays unchanged

        # vs_idx: z-index of PSS output (0 if no PSS)
        old_vs = Int(table.vs_idx[k])
        if old_vs > 0 && old_vs <= n_da
            table.vs_idx[k] = Int32(old_to_new[old_vs])
        end
    end
    return nothing
end

function _remap_esdc1a!(table::ESDC1ATable, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_dp = Int(table.diff_ptr[k])
        table.diff_ptr[k] = Int32(old_to_new[old_dp])

        # vr_idx is network voltage, stays unchanged

        old_vs = Int(table.vs_idx[k])
        if old_vs > 0 && old_vs <= n_da
            table.vs_idx[k] = Int32(old_to_new[old_vs])
        end
    end
    return nothing
end

function _remap_ieeest!(table::IEEESTTable, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_dp = Int(table.diff_ptr[k])
        table.diff_ptr[k] = Int32(old_to_new[old_dp])

        old_ap_global = Int(table.alg_ptr[k]) + diff_dim
        table.alg_ptr[k] = Int32(old_to_new[old_ap_global] - diff_dim)

        # w_idx: z-index of generator's omega
        old_w = Int(table.w_idx[k])
        if old_w > 0 && old_w <= n_da
            table.w_idx[k] = Int32(old_to_new[old_w])
        end
    end
    return nothing
end

function _remap_static_gen!(table::StaticGenTable, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for k in 1:table.n
        old_ap = Int(table.alg_ptr[k])
        if old_ap > 0 && old_ap <= n_da
            table.alg_ptr[k] = Int32(old_to_new[old_ap])
        end
        # vr_idx is network voltage, stays unchanged
    end
    return nothing
end

# -----------------------------------------------------------------------
# A_k / B_k / C_k extraction helpers (D3)
# -----------------------------------------------------------------------

"""
    extract_Ak!(A_dense, J_sparse, cluster::DeviceCluster)

Copy the |w_k| x |w_k| block from the global sparse Jacobian J_sparse
into A_dense. The block corresponds to rows/cols w_start:w_end of the
cluster.
"""
function extract_Ak!(A_dense::AbstractMatrix, J_sparse::SparseMatrixCSC,
                      cluster::DeviceCluster)
    ws = cluster.w_start
    we = cluster.w_end
    wk = cluster.w_size
    @assert size(A_dense, 1) >= wk && size(A_dense, 2) >= wk

    rows = rowvals(J_sparse)
    vals = nonzeros(J_sparse)

    fill!(A_dense, 0.0)
    @inbounds for col_global in ws:we
        local_col = col_global - ws + 1
        for nz in nzrange(J_sparse, col_global)
            row_global = rows[nz]
            if ws <= row_global <= we
                local_row = row_global - ws + 1
                A_dense[local_row, local_col] = vals[nz]
            end
        end
    end
    return nothing
end

"""
    extract_Bk_Ck!(B_tilde, C_tilde, J_sparse, cluster::DeviceCluster, net_ptr::Int)

Extract the rank-2 coupling pieces:
  B_tilde in R^{|w_k| x 2}: columns of J at (w_k rows, vr/vi columns)
  C_tilde in R^{2 x |w_k|}: rows of J at (vr/vi rows, w_k columns)

where vr = net_ptr + 2*(bus-1) + 1, vi = vr + 1.
"""
function extract_Bk_Ck!(B_tilde::AbstractMatrix, C_tilde::AbstractMatrix,
                          J_sparse::SparseMatrixCSC,
                          cluster::DeviceCluster, net_ptr::Int)
    ws = cluster.w_start
    we = cluster.w_end
    wk = cluster.w_size
    bus = cluster.bus

    vr_col = net_ptr + 2*(bus - 1) + 1
    vi_col = vr_col + 1

    @assert size(B_tilde, 1) >= wk && size(B_tilde, 2) >= 2
    @assert size(C_tilde, 1) >= 2  && size(C_tilde, 2) >= wk

    fill!(B_tilde, 0.0)
    fill!(C_tilde, 0.0)

    rows = rowvals(J_sparse)
    vals = nonzeros(J_sparse)

    # B_tilde: for each voltage column (vr, vi), extract the w_k rows
    for (lcol, gcol) in enumerate((vr_col, vi_col))
        for nz in nzrange(J_sparse, gcol)
            row_global = rows[nz]
            if ws <= row_global <= we
                local_row = row_global - ws + 1
                B_tilde[local_row, lcol] = vals[nz]
            end
        end
    end

    # C_tilde: for each w_k column, extract the vr/vi rows
    for col_global in ws:we
        local_col = col_global - ws + 1
        for nz in nzrange(J_sparse, col_global)
            row_global = rows[nz]
            if row_global == vr_col
                C_tilde[1, local_col] = vals[nz]
            elseif row_global == vi_col
                C_tilde[2, local_col] = vals[nz]
            end
        end
    end
    return nothing
end
