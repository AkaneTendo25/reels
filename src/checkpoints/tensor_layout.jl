@enum TensorLayout begin
    SCALAR_LAYOUT
    VECTOR_LAYOUT
    LINEAR_OUT_IN
    CONV_OUT_IN_SPATIAL
    ROW_MAJOR_SOURCE
    NATIVE_LAYOUT
end

"""
Restore a SafeTensors row-major tensor to its documented source-framework axes.
`load_safetensor` exposes the zero-copy reversed-axis view; all required copies
and permutations are centralized here rather than scattered through forwards.
"""
function restore_source_layout(a::AbstractArray, layout::TensorLayout)
    layout in (SCALAR_LAYOUT, VECTOR_LAYOUT, NATIVE_LAYOUT) && return copy(a)
    permutedims(a, reverse(1:ndims(a)))
end
