isempty(ARGS) && error("missing Julia entrypoint")
entrypoint = abspath(popfirst!(ARGS))
if Sys.islinux()
    process_name = get(ENV, "REELS_PROCESS_NAME", "reels-julia")
    isempty(process_name) && error("REELS_PROCESS_NAME must not be empty")
    ncodeunits(process_name) <= 15 ||
        error("REELS_PROCESS_NAME must fit Linux's 15-byte comm limit")
    status = ccall(:prctl, Cint,
        (Cint, Cstring, Culong, Culong, Culong),
        15, process_name, 0, 0, 0)
    status == 0 || error("failed to set process name")
    strip(read("/proc/self/comm", String)) == process_name ||
        error("process name did not update")
end
include(entrypoint)
