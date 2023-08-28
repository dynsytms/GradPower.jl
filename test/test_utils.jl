A = sparse([1 0 2; 0 3 0; 4 0 0; 0 5 0; 0 0 6])
row_map = GradPower.create_row_to_pos_array(A)  # Assuming you have this function

# Test csc_set_row!
@testset "csc_set_row!" begin
    B = copy(A)
    GradPower.csc_set_row!(B, row_map, 1, [1, 3], [10, 20])
    @test B[1, 1] == 10
    @test B[1, 3] == 20
    @test B[1, 2] == 0
end

# Test csc_add_row!
@testset "csc_add_row!" begin
    B = copy(A)
    GradPower.csc_add_row!(B, row_map, 1, [1, 3], [1, 1])
    @test B[1, 1] == 2
    @test B[1, 3] == 3
    @test B[1, 2] == 0
end

# Test csc_mult_row!
@testset "csc_mult_row!" begin
    B = copy(A)
    GradPower.csc_mult_row!(B, row_map, 1, 2)
    @test B[1, 1] == 2
    @test B[1, 3] == 4
    @test B[1, 2] == 0
end
