mutable struct BusRaw
    busn::Int64
    name::String
    baseKV::Float64
    type::Int64
    area::Int64
    zone::Int64
    owner::Int64
    vm::Float64
    va::Float64
end

function BusRaw(line::String)
    line = split(strip(line, '\n'), ',')
    busn = parse(Int64, line[1])
    name = line[2]
    baseKV = parse(Float64, line[3])
    type = parse(Int64, line[4])
    area = parse(Int64, line[5])
    zone = parse(Int64, line[6])
    owner = parse(Int64, line[7])
    vm = parse(Float64, line[8])
    va = parse(Float64, line[9])
    busraw = BusRaw(busn, name, baseKV, type, area, zone, owner, vm, va)
    return busraw
end

mutable struct LoadRaw
    busn::Int64
    name::String
    status::Int64
    area::Int64
    zone::Int64
    pl::Float64
    ql::Float64
    ip::Float64
    iq::Float64
    yp::Float64
    yq::Float64
    owner::Int64
end

function LoadRaw(line::String)
    line = split(strip(line, '\n'), ',')
    busn = parse(Int64, line[1])
    name = strip(line[2], '\'')
    status = parse(Int64, line[3])
    area = parse(Int64, line[4])
    zone = parse(Int64, line[5])
    pl = parse(Float64, line[6])
    ql = parse(Float64, line[7])
    ip = parse(Float64, line[8])
    iq = parse(Float64, line[9])
    yp = parse(Float64, line[10])
    yq = parse(Float64, line[11])
    owner = parse(Int64, line[12])
    loadraw = LoadRaw(busn, name, status, area, zone, pl, ql, ip, iq, yp, yq, owner)
    return loadraw
end

mutable struct ShuntRaw
    busn::Int64
    name::String
    status::Int64
    gshunt::Float64
    bshunt::Float64
end

function ShuntRaw(line::String)
    line = split(strip(line, '\n'), ',')
    busn = parse(Int64, line[1])
    name = line[2]
    status = parse(Int64, line[3])
    gshunt = parse(Float64, line[4])
    bshunt = parse(Float64, line[5])
    shuntraw = ShuntRaw(busn, name, status, gshunt, bshunt)
    return shuntraw
end

mutable struct GenRaw
    busn::Int64
    name::String
    pg::Float64
    qg::Float64
    qt::Float64
    qb::Float64
    vs::Float64
    ireg::Int64
    mbase::Float64
    zr::Float64
    zx::Float64
    rt::Float64
    xt::Float64
    gtap::Float64
    status::Int64
    rmpct::Float64
    pt::Float64
    pb::Float64
    o1::Int64
    f1::Float64
end

function GenRaw(line::String)
    line = split(strip(line, '\n'), ',')
    busn = parse(Int64, line[1])
    name = line[2]
    pg = parse(Float64, line[3])
    qg = parse(Float64, line[4])
    qt = parse(Float64, line[5])
    qb = parse(Float64, line[6])
    vs = parse(Float64, line[7])
    ireg = parse(Int64, line[8])
    mbase = parse(Float64, line[9])
    zr = parse(Float64, line[10])
    zx = parse(Float64, line[11])
    rt = parse(Float64, line[12])
    xt = parse(Float64, line[13])
    gtap = parse(Float64, line[14])
    status = parse(Int64, line[15])
    rmpct = parse(Float64, line[16])
    pt = parse(Float64, line[17])
    pb = parse(Float64, line[18])
    o1 = parse(Int64, line[19])
    f1 = parse(Float64, line[20])
    genraw = GenRaw(busn, name, pg, qg, qt, qb, vs, ireg, mbase, zr, zx, rt, xt, gtap, status, rmpct, pt, pb, o1, f1)
    return genraw
end

mutable struct BranchRaw
    fbus::Int64
    tbus::Int64
    ckt::String
    r::Float64
    x::Float64
    b::Float64
    rateA::Float64
    rateB::Float64
    rateC::Float64
    gi::Float64
    bi::Float64
    gj::Float64
    bj::Float64
    status::Int64
    MET::Int64
    length::Float64
    o1::Int64
    f1::Float64
end

function BranchRaw(line::String)
    line = split(strip(line, '\n'), ',')
    fbus = parse(Int64, line[1])
    tbus = parse(Int64, line[2])
    ckt = line[3]
    r = parse(Float64, line[4])
    x = parse(Float64, line[5])
    b = parse(Float64, line[6])
    rateA = parse(Float64, line[7])
    rateB = parse(Float64, line[8])
    rateC = parse(Float64, line[9])
    gi = parse(Float64, line[10])
    bi = parse(Float64, line[11])
    gj = parse(Float64, line[12])
    bj = parse(Float64, line[13])
    status = parse(Int64, line[14])
    MET = parse(Int64, line[15])
    length = parse(Float64, line[16])
    o1 = parse(Int64, line[17])
    f1 = parse(Float64, line[18])
    branchraw = BranchRaw(fbus, tbus, ckt, r, x, b, rateA, rateB, rateC, gi, bi, gj, bj, status, MET, length, o1, f1)
    return branchraw
end

mutable struct TransformerRaw
    fbus::Int64
    tbus::Int64
    k::Int64
    ckt::String
    CW::Int64
    CZ::Int64
    CM::Int64
    MAG1::Float64
    MAG2::Float64
    NMETR::Int64
    tname::String
    status::Int64
    o1::Float64
    r::Float64
    x::Float64
    sbase12::Float64
    WINDV1::Float64
    NOMV1::Float64
    ANG1::Float64
    rateA::Float64
    rateB::Float64
    rateC::Float64
    COD1::Int64
    CONT1::Int64
    RMA1::Float64
    RMIT::Float64
    VMA1::Float64
    VMI1::Float64
    NTP1::Int64
    TAB1::Int64
    CR1::Float64
    CX1::Float64
    CNXA1::Float64
    WINDV2::Float64
    NOMV2::Float64
end

function TransformerRaw(line::String, line2::String, line3::String, line4::String)
    line = split(strip(line, '\n'), ',')
    fbus = parse(Int64, line[1])
    tbus = parse(Int64, line[2])
    k = parse(Int64, line[3])
    ckt = line[4]
    CW = parse(Int64, line[5])
    CZ = parse(Int64, line[6])
    CM = parse(Int64, line[7])
    MAG1 = parse(Float64, line[8])
    MAG2 = parse(Float64, line[9])
    NMETR = parse(Int64, line[10])
    tname = line[11]
    status = parse(Int64, line[12])
    o1 = parse(Float64, line[13])
    line2 = split(strip(line2, '\n'), ',')
    r = parse(Float64, line2[1])
    x = parse(Float64, line2[2])
    sbase12 = parse(Float64, line2[3])
    line3 = split(strip(line3, '\n'), ',')
    WINDV1 = parse(Float64, line3[1])
    NOMV1 = parse(Float64, line3[2])
    ANG1 = parse(Float64, line3[3])
    rateA = parse(Float64, line3[4])
    rateB = parse(Float64, line3[5])
    rateC = parse(Float64, line3[6])
    COD1 = parse(Int64, line3[7])
    CONT1 = parse(Int64, line3[8])
    RMA1 = parse(Float64, line3[9])
    RMIT = parse(Float64, line3[10])
    VMA1 = parse(Float64, line3[11])
    VMI1 = parse(Float64, line3[12])
    NTP1 = parse(Int64, line3[13])
    TAB1 = parse(Int64, line3[14])
    CR1 = parse(Float64, line3[15])
    CX1 = parse(Float64, line3[16])
    CNXA1 = parse(Float64, line3[17])
    line4 = split(strip(line4, '\n'), ',')
    WINDV2 = parse(Float64, line4[1])
    NOMV2 = parse(Float64, line4[2])
    transraw = TransformerRaw(fbus, tbus, k, ckt, CW, CZ, CM, MAG1, MAG2, NMETR, tname, status, o1, r, x, sbase12, WINDV1, NOMV1, ANG1, rateA, rateB, rateC, COD1, CONT1, RMA1, RMIT, VMA1, VMI1, NTP1, TAB1, CR1, CX1, CNXA1, WINDV2, NOMV2)
    return transraw
end

mutable struct ThreeTransformerRaw
    ibus::Int64
    jbus::Int64
    kbus::Int64
    ckt::String
    CW::Int64
    CZ::Int64
    CM::Int64
    MAG1::Float64
    MAG2::Float64
    NMETR::Int64
    tname::String
    status::Int64
    o1::Float64
    r12::Float64
    x12::Float64
    sbase12::Float64
    r23::Float64
    x23::Float64
    sbase23::Float64
    r13::Float64
    x13::Float64
    sbase31::Float64
    vmstar::Float64
    anstar::Float64
    WINDV1::Float64
    NOMV1::Float64
    ANG1::Float64
    rateA1::Float64
    rateB1::Float64
    rateC1::Float64
    COD1::Int64
    CONT1::Int64
    RMA1::Float64
    RMI1::Float64
    VMA1::Float64
    VMI1::Float64
    NTP1::Int64
    TAB1::Int64
    CR1::Float64
    CX1::Float64
    CNXA1::Float64
    WINDV2::Float64
    NOMV2::Float64
    ANG2::Float64
    rateA2::Float64
    rateB2::Float64
    rateC2::Float64
    COD2::Int64
    CONT2::Int64
    RMA2::Float64
    RMI2::Float64
    VMA2::Float64
    VMI2::Float64
    NTP2::Int64
    TAB2::Int64
    CR2::Float64
    CX2::Float64
    CNXA2::Float64
    WINDV3::Float64
    NOMV3::Float64
    ANG3::Float64
    rateA3::Float64
    rateB3::Float64
    rateC3::Float64
    COD3::Int64
    CONT3::Int64
    RMA3::Float64
    RMI3::Float64
    VMA3::Float64
    VMI3::Float64
    NTP3::Int64
    TAB3::Int64
    CR3::Float64
    CX3::Float64
    CNXA3::Float64
end

function ThreeTransformerRaw(line::String, line2::String, line3::String, line4::String, line5::String)
    line = split(strip(line, '\n'), ',')
    ibus = parse(Int64, line[1])
    jbus = parse(Int64, line[2])
    kbus = parse(Int64, line[3])
    ckt = line[4]
    CW = parse(Int64, line[5])
    CZ = parse(Int64, line[6])
    CM = parse(Int64, line[7])
    MAG1 = parse(Float64, line[8])
    MAG2 = parse(Float64, line[9])
    NMETR = parse(Int64, line[10])
    tname = line[11]
    status = parse(Int64, line[12])
    o1 = parse(Float64, line[13])
    line2 = split(strip(line2, '\n'), ',')
    r12 = parse(Float64, line2[1])
    x12 = parse(Float64, line2[2])
    sbase12 = parse(Float64, line2[3])
    r23 = parse(Float64, line2[4])
    x23 = parse(Float64, line2[5])
    sbase23 = parse(Float64, line2[6])
    r13 = parse(Float64, line2[7])
    x13 = parse(Float64, line2[8])
    sbase31 = parse(Float64, line2[9])
    vmstar = parse(Float64, line2[10])
    anstar = parse(Float64, line2[11])
    line3 = split(strip(line3, '\n'), ',')
    WINDV1 = parse(Float64, line3[1])
    NOMV1 = parse(Float64, line3[2])
    ANG1 = parse(Float64, line3[3])
    rateA1 = parse(Float64, line3[4])
    rateB1 = parse(Float64, line3[5])
    rateC1 = parse(Float64, line3[6])
    COD1 = parse(Int64, line3[7])
    CONT1 = parse(Int64, line3[8])
    RMA1 = parse(Float64, line3[9])
    RMI1 = parse(Float64, line3[10])
    VMA1 = parse(Float64, line3[11])
    VMI1 = parse(Float64, line3[12])
    NTP1 = parse(Int64, line3[13])
    TAB1 = parse(Int64, line3[14])
    CR1 = parse(Float64, line3[15])
    CX1 = parse(Float64, line3[16])
    CNXA1 = parse(Float64, line3[17])
    line4 = split(strip(line4, '\n'), ',')
    WINDV2 = parse(Float64, line4[1])
    NOMV2 = parse(Float64, line4[2])
    ANG2 = parse(Float64, line4[3])
    rateA2 = parse(Float64, line4[4])
    rateB2 = parse(Float64, line4[5])
    rateC2 = parse(Float64, line4[6])
    COD2 = parse(Int64, line4[7])
    CONT2 = parse(Int64, line4[8])
    RMA2 = parse(Float64, line4[9])
    RMI2 = parse(Float64, line4[10])
    VMA2 = parse(Float64, line4[11])
    VMI2 = parse(Float64, line4[12])
    NTP2 = parse(Int64, line4[13])
    TAB2 = parse(Int64, line4[14])
    CR2 = parse(Float64, line4[15])
    CX2 = parse(Float64, line4[16])
    CNXA2 = parse(Float64, line4[17])
    line5 = split(strip(line5, '\n'), ',')
    WINDV3 = parse(Float64, line5[1])
    NOMV3 = parse(Float64, line5[2])
    ANG3 = parse(Float64, line5[3])
    rateA3 = parse(Float64, line5[4])
    rateB3 = parse(Float64, line5[5])
    rateC3 = parse(Float64, line5[6])
    COD3 = parse(Int64, line5[7])
    CONT3 = parse(Int64, line5[8])
    RMA3 = parse(Float64, line5[9])
    RMI3 = parse(Float64, line5[10])
    VMA3 = parse(Float64, line5[11])
    VMI3 = parse(Float64, line5[12])
    NTP3 = parse(Int64, line5[13])
    TAB3 = parse(Int64, line5[14])
    CR3 = parse(Float64, line5[15])
    CX3 = parse(Float64, line5[16])
    CNXA3 = parse(Float64, line5[17])
    transraw = ThreeTransformerRaw(ibus, jbus, kbus, ckt, CW, CZ, CM, MAG1, MAG2, NMETR, tname, status, o1, r12, x12, sbase12, r23, x23, sbase23, r13, x13, sbase31, vmstar, anstar, WINDV1, NOMV1, ANG1, rateA1, rateB1, rateC1, COD1, CONT1, RMA1, RMI1, VMA1, VMI1, NTP1, TAB1, CR1, CX1, CNXA1, WINDV2, NOMV2, ANG2, rateA2, rateB2, rateC2, COD2, CONT2, RMA2, RMI2, VMA2, VMI2, NTP2, TAB2, CR2, CX2, CNXA2, WINDV3, NOMV3, ANG3, rateA3, rateB3, rateC3, COD3, CONT3, RMA3, RMI3, VMA3, VMI3, NTP3, TAB3, CR3, CX3, CNXA3)
    return transraw
end

mutable struct SwitchedShuntRaw
    busn::Int64
    status::Int64
    binit::Float64
end

function SwitchedShuntRaw(line::String)
    parts = split(strip(line, '\n'), ',')
    busn = parse(Int64, parts[1])
    status = length(parts) > 3 ? parse(Int64, parts[4]) : 0
    binit = length(parts) > 9 ? parse(Float64, parts[10]) : 0.0
    return SwitchedShuntRaw(busn, status, binit)
end

mutable struct PsystemRaw
    baseMVA::Float64
    buses::Array{BusRaw, 1}
    loads::Array{LoadRaw, 1}
    shunts::Array{ShuntRaw, 1}
    branches::Array{BranchRaw, 1}
    gens::Array{GenRaw, 1}
    transformers::Array{TransformerRaw, 1}
    transthree::Array{ThreeTransformerRaw, 1}
    switched_shunts::Array{SwitchedShuntRaw, 1}
end

function PsystemRaw(line::String)
    baseMVA = parse(Float64, split(strip(line[1:12]), ',')[2])
    buses = []
    loads = []
    shunts = []
    branches = []
    gens = []
    transformers = []
    transthree = []
    switched_shunts = []
    sys = PsystemRaw(baseMVA, buses, loads, shunts, branches, gens, transformers, transthree, switched_shunts)
    return sys
end

function add_buses!(sys::PsystemRaw, f::IOStream)
    while true
        line = readline(f)
        if occursin("0 / END OF BUS DATA, BEGIN LOAD DATA", line)
            break
        elseif isempty(line)
            break
        else
            push!(sys.buses, BusRaw(line))
        end
    end
end

function add_loads!(sys::PsystemRaw, f::IOStream)
    while true
        line = readline(f)
        if occursin("0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA", line)
            break
        elseif isempty(line)
            break
        else
            push!(sys.loads, LoadRaw(line))
        end
    end
end

function add_shunts!(sys::PsystemRaw, f::IOStream)
    while true
        line = readline(f)
        if occursin("0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA", line)
            break
        elseif isempty(line)
            break
        else
            push!(sys.shunts, ShuntRaw(line))
        end
    end
end

function add_gens!(sys::PsystemRaw, f::IOStream)
    while true
        line = readline(f)
        if occursin("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA", line)
            break
        elseif isempty(line)
            break
        else
            push!(sys.gens, GenRaw(line))
        end
    end
end

function add_branches!(sys::PsystemRaw, f::IOStream)
    while true
        line = readline(f)
        if occursin("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA", line)
            break
        elseif isempty(line)
            break
        else
            push!(sys.branches, BranchRaw(line))
        end
    end
end

function add_transformers!(sys::PsystemRaw, f::IOStream)
    while true
        line = readline(f)
        if occursin("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA", line)
            break
        elseif isempty(line)
            break
        else
            line_trans = split(strip(line, '\n'), ',')
            kbus = parse(Int64, line_trans[3])
            if kbus == 0
                line2 = readline(f)
                line3 = readline(f)
                line4 = readline(f)
                push!(sys.transformers, TransformerRaw(line, line2, line3, line4))
            else
                line2 = readline(f)
                line3 = readline(f)
                line4 = readline(f)
                line5 = readline(f)
                push!(sys.transthree, ThreeTransformerRaw(line, line2, line3, line4, line5))
            end
        end
    end
end

function add_switched_shunts!(sys::PsystemRaw, f::IOStream)
    # Skip section headers (areas, DC links, FACTS, etc.) until we hit switched shunts or EOF.
    while !eof(f)
        line = readline(f)
        if occursin("BEGIN SWITCHED SHUNT DATA", line)
            break
        end
    end
    while !eof(f)
        line = readline(f)
        if occursin("END OF SWITCHED SHUNT DATA", line)
            break
        elseif isempty(strip(line))
            continue
        else
            push!(sys.switched_shunts, SwitchedShuntRaw(line))
        end
    end
end

function read_psse_raw(filename::String)
    f = open(filename, "r")
    sys = PsystemRaw(readline(f))
    readline(f)  # skip two spaces
    readline(f)
    add_buses!(sys, f)
    add_loads!(sys, f)
    add_shunts!(sys, f)
    add_gens!(sys, f)
    add_branches!(sys, f)
    add_transformers!(sys, f)
    add_switched_shunts!(sys, f)
    close(f)
    return sys
end
