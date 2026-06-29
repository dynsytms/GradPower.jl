function _sizes_of(dtype_instance)
    return (Int(dtype_instance.diff_size),
            Int(dtype_instance.alg_size),
            Int(dtype_instance.ctrl_size),
            Int(dtype_instance.par_size))
end

let
    g = GenericGenerator(1, "1")
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :genrou, Genrou, :generator,
        d, a, c, p,
        :genrou_residual_batch!, :genrou_jacobian_batch!,
        nothing,
        :none,
    ))
end

let
    g = IEESGO(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :ieesgo, IEESGO, :governor,
        d, a, c, p,
        :ieesgo_residual_batch!, :ieesgo_jacobian_batch!,
        nothing,
        :genrou,
    ))
end

let
    g = TGOV1(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :tgov1, TGOV1, :governor,
        d, a, c, p,
        :tgov1_residual_batch!, :tgov1_jacobian_batch!,
        nothing,
        :genrou,
    ))
end

let
    g = SEXS(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :sexs, SEXS, :exciter,
        d, a, c, p,
        :sexs_residual_batch!, :sexs_jacobian_batch!,
        nothing,
        :genrou,
    ))
end

let
    g = ESDC1A(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :esdc1a, ESDC1A, :exciter,
        d, a, c, p,
        :esdc1a_residual_batch!, :esdc1a_jacobian_batch!,
        nothing,
        :genrou,
    ))
end

let
    g = IEEEST(1, "1", 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 999.0, -999.0, 999.0, -999.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :ieeest, IEEEST, :stabilizer,
        d, a, c, p,
        :ieeest_residual_batch!, :ieeest_jacobian_batch!,
        nothing,
        :sexs,
    ))
end

let
    g = ZIPLoad(1, "1", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :zipload, ZIPLoad, :load,
        d, a, c, p,
        :zipload_residual_batch!, :zipload_jacobian_batch!,
        nothing,
        :none,
    ))
end

let
    g = StaticGenerator(Int64(1), Int64(2), Int64[1], 1.0, 0.0)
    d, a, c, p = _sizes_of(g)
    register_contract!(DeviceContract(
        :static_gen, StaticGenerator, :static_gen,
        d, a, c, p,
        :static_gen_residual_batch!, :static_gen_jacobian_batch!,
        nothing,
        :none,
    ))
end

# GENSAL is a parameter-substituted Genrou (x_qp = x_dp, T_q0p = T_d0p);
# the .dyr parser already materializes it as a Genrou instance. Register
# as an alias so registry consumers can resolve :gensal -> the Genrou
# kernel/contract without a duplicated entry.
register_alias!(:gensal, :genrou)
