module Constants

# === Exported Symbols ===
export GevInvTofm, fmGeV

"""
    GevInvTofm

Conversion factor from inverse GeV (natural units) to femtometers (fm).

1 [GeV⁻¹] ≈ 0.1975 [fm], so:
"""
const GevInvTofm = 1 / 5.068  # ≈ 0.1975 fm

"""
    fmGeV

Conversion factor from femtometers (fm) to GeV³.

Useful when expressing densities (e.g. n(r, t)) in [1/fm³] using natural units.
"""
const fmGeV = 1 / GevInvTofm  # ≈ 5.068 GeV

end # module Constants
