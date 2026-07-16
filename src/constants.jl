module Constants

# === Exported Symbols ===
export GevInvTofm, fmGeV

"""
    GevInvTofm

Conversion factor from inverse GeV (natural units) to femtometers (fm).

1 [GeV⁻¹] ≈ 0.1975 [fm], so:
"""
const GevInvTofm = 0.197327  # ħc in GeV·fm (PDG). Standardized from the legacy 1/5.068=0.197316,
                             # which caused a 5.3e-5 τ_n inconsistency vs FokkerPlank2D (audit A6-1).

"""
    fmGeV

Conversion factor from femtometers (fm) to GeV³.

Useful when expressing densities (e.g. n(r, t)) in [1/fm³] using natural units.
"""
const fmGeV = 1 / GevInvTofm  # ≈ 5.068 GeV

end # module Constants
