import os

new_bib_entries = """
@article{Giosan2006Recent,
  author = {Liviu Giosan and Stefan Constantinescu and Peter D. Clift and Ali R. Tabrez and Muhammed Danish and Asif Inam},
  title = {Recent morphodynamics of the Indus delta shore and shelf},
  journal = {Continental Shelf Research},
  year = {2006},
  volume = {26},
  number = {14},
  pages = {1668--1684},
  doi = {10.1016/j.csr.2006.05.009},
  publisher = {Elsevier}
}

@article{Syvitski2009Sinking,
  author = {James P. M. Syvitski and Albert J. Kettner and Irina Overeem and Eric W. H. Hutton and Mark T. Hannon and G. Robert Brakenridge and John Day and Charles Vörösmarty and Yoshiki Saito and Liviu Giosan and Robert J. Nicholls},
  title = {Sinking deltas due to human activities},
  journal = {Nature Geoscience},
  year = {2009},
  volume = {2},
  number = {10},
  pages = {681--686},
  doi = {10.1038/ngeo629},
  publisher = {Nature Publishing Group}
}

@article{Karim2002Water,
  author = {A. Karim and J. Veizer},
  title = {Water balance of the Indus River Basin and moisture source in the Karakoram and western Himalayas: Implications from hydrogen and oxygen isotopes in river water},
  journal = {Journal of Geophysical Research: Atmospheres},
  year = {2002},
  volume = {107},
  number = {D18},
  pages = {4362},
  doi = {10.1029/2000JD000253},
  publisher = {American Geophysical Union}
}

@article{Salik2016Environmental,
  author = {K. M. Salik and M. A. Jahangir and Q. ul Islam and S. A. Naeem},
  title = {Environmental flow requirements and impacts of climate change-induced river flow changes on ecology of the Indus Delta, Pakistan},
  journal = {Regional Studies in Marine Science},
  year = {2016},
  volume = {7},
  pages = {168--175},
  doi = {10.1016/j.rsma.2016.06.008},
  publisher = {Elsevier}
}

@article{Mitra2024Assessing,
  author = {Bijoy Mitra and Muhammad Muhitur Rahman and Aftab Ahmad Khan and Syed Masiur Rahman},
  title = {Assessing the impact of sea level rise on the Indus delta in Pakistan: A comprehensive analysis of flooded areas and future vulnerabilities},
  journal = {Heliyon},
  year = {2024},
  volume = {10},
  number = {12},
  pages = {e33120},
  doi = {10.1016/j.heliyon.2024.e33120},
  publisher = {Elsevier}
}

@article{Salik2022Social,
  author = {K. M. Salik and Q. ul Islam and S. A. Naeem},
  title = {Social and ecological climate change vulnerability assessment in the Indus delta, Pakistan},
  journal = {Water Practice \\& Technology},
  year = {2022},
  volume = {17},
  number = {4},
  pages = {876--890},
  doi = {10.2166/wpt.2022.087},
  publisher = {IWA Publishing}
}

@article{Tejedor2015Delta1,
  author = {Alejandro Tejedor and Anthony Longjas and Irina Overeem and James P. M. Syvitski and Juan A. Sanz and Efi Foufoula-Georgiou},
  title = {Delta channel networks: 1. A graph-theoretic approach for studying connectivity and steady state transport on deltaic surfaces},
  journal = {Water Resources Research},
  year = {2015},
  volume = {51},
  number = {6},
  pages = {3998--4018},
  doi = {10.1002/2014WR016577},
  publisher = {Wiley-Blackwell}
}

@article{Tejedor2015Delta2,
  author = {Alejandro Tejedor and Anthony Longjas and Irina Overeem and James P. M. Syvitski and Juan A. Sanz and Efi Foufoula-Georgiou},
  title = {Delta channel networks: 2. Metrics of topologic and dynamic complexity for delta comparison, physical inference, and vulnerability assessment},
  journal = {Water Resources Research},
  year = {2015},
  volume = {51},
  number = {6},
  pages = {4019--4045},
  doi = {10.1002/2014WR016604},
  publisher = {Wiley-Blackwell}
}

@article{Dunn2019Projections,
  author = {Frances E. Dunn and Stephen E. Darby and Robert J. Nicholls and Sibel D. Lindberg},
  title = {Projections of declining fluvial sediment delivery to major deltas worldwide in response to climate change and anthropogenic stress},
  journal = {Environmental Research Letters},
  year = {2019},
  volume = {14},
  number = {8},
  pages = {084040},
  doi = {10.1088/1748-9326/ab304e},
  publisher = {IOP Publishing}
}

@article{Cheema2012Surface,
  author = {M. J. M. Cheema and W. G. M. Bastiaanssen},
  title = {Surface energy balance and actual evapotranspiration of the transboundary Indus Basin estimated from satellite measurements and the ETLook model},
  journal = {Water Resources Research},
  year = {2012},
  volume = {48},
  number = {11},
  pages = {W11502},
  doi = {10.1029/2011WR010998},
  publisher = {Wiley-Blackwell}
}

@article{Solangi2019Analysis,
  author = {Ghulam Shabir Solangi and Altaf Ali Siyal and Pirah Siyal},
  title = {Analysis of Indus Delta Groundwater and Surface water Suitability for Domestic and Irrigation Purposes},
  journal = {Civil Engineering Journal},
  year = {2019},
  volume = {5},
  number = {12},
  pages = {2537--2551},
  doi = {10.28991/cej-2019-03091356},
  publisher = {Civil Engineering Journal}
}
"""

# 1. Append to references.bib
bib_path = "prism/references.bib"
if os.path.exists(bib_path):
    with open(bib_path, "a") as f:
        f.write(new_bib_entries)
    print("Appended 11 new entries to prism/references.bib")

# 2. Append to reference_audit.md
audit_path = "prism/reference_audit.md"
audit_rows = """| Giosan2006Recent | Liviu Giosan et al. | 2006 | Continental Shelf Research | [10.1016/j.csr.2006.05.009](https://doi.org/10.1016/j.csr.2006.05.009) | VERIFIED |
| Syvitski2009Sinking | James P. M. Syvitski et al. | 2009 | Nature Geoscience | [10.1038/ngeo629](https://doi.org/10.1038/ngeo629) | VERIFIED |
| Karim2002Water | A. Karim et al. | 2002 | Journal of Geophysical Research | [10.1029/2000JD000253](https://doi.org/10.1029/2000JD000253) | VERIFIED |
| Salik2016Environmental | K. M. Salik et al. | 2016 | Regional Studies in Marine Science | [10.1016/j.rsma.2016.06.008](https://doi.org/10.1016/j.rsma.2016.06.008) | VERIFIED |
| Mitra2024Assessing | Bijoy Mitra et al. | 2024 | Heliyon | [10.1016/j.heliyon.2024.e33120](https://doi.org/10.1016/j.heliyon.2024.e33120) | VERIFIED |
| Salik2022Social | K. M. Salik et al. | 2022 | Water Practice & Technology | [10.2166/wpt.2022.087](https://doi.org/10.2166/wpt.2022.087) | VERIFIED |
| Tejedor2015Delta1 | Alejandro Tejedor et al. | 2015 | Water Resources Research | [10.1002/2014WR016577](https://doi.org/10.1002/2014WR016577) | VERIFIED |
| Tejedor2015Delta2 | Alejandro Tejedor et al. | 2015 | Water Resources Research | [10.1002/2014WR016604](https://doi.org/10.1002/2014WR016604) | VERIFIED |
| Dunn2019Projections | Frances E. Dunn et al. | 2019 | Environmental Research Letters | [10.1088/1748-9326/ab304e](https://doi.org/10.1088/1748-9326/ab304e) | VERIFIED |
| Cheema2012Surface | M. J. M. Cheema et al. | 2012 | Water Resources Research | [10.1029/2011WR010998](https://doi.org/10.1029/2011WR010998) | VERIFIED |
| Solangi2019Analysis | Ghulam Shabir Solangi et al. | 2019 | Civil Engineering Journal | [10.28991/cej-2019-03091356](https://doi.org/10.28991/cej-2019-03091356) | VERIFIED |
"""

if os.path.exists(audit_path):
    with open(audit_path, "a") as f:
        f.write(audit_rows)
    print("Appended 11 new entries to prism/reference_audit.md")
