using Libdl

source = joinpath(@__DIR__, "sentencepiece_bridge.cpp")
libdir = joinpath(@__DIR__, "usr", "lib")
mkpath(libdir)
library = joinpath(libdir, "libreels_sentencepiece." * Libdl.dlext)

cxx = get(ENV, "CXX", "c++")
command = `$cxx -std=c++17 -O2 -fPIC -shared $source -lsentencepiece -o $library`
@info "building native SentencePiece bridge" command library
run(command)
