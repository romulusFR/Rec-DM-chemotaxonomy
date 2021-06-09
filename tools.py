"""Some tools"""

# %%
import doctest
import csv


def replace_greek_alphabet(to_translate: str) -> str:
    """Replace Greek letter by their full latin name

    Parameters
    ----------
    to_translate : str
        the string to transliterate

    Returns
    -------
    str
        the string to translate with all occurences of greel letter replaced

    Note
    ----
    works only for lower case Greek letter

    Examples
    --------

    >>> remove_greek_alphabet("α-pinene")
    'alpha-pinene'
    >>> remove_greek_alphabet("β-pinene")
    'beta-pinene'
    >>> remove_greek_alphabet("From α to ω")
    'From alpha to omega'
    """

    greek_to_latin = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "ι": "iota",
        "κ": "kappa",
        "λ": "lamda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "ο": "omicron",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "ς": "sigma",  # final final ?
        "τ": "tau",
        "υ": "upsilon",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
    }
    res = to_translate
    for letter, latin in greek_to_latin.items():
        res = res.replace(letter, latin)
    return res


SAMPLE_KEGG_RESULT = """cpd:C00521\t(S)-Limonene; (-)-Limonene; (4S)-1-Methyl-4-(prop-1-en-2-yl)cyclohexene; (-)-(S)-Limonene; (-)-(4S)-Limonene\ncpd:C06078\tLimonene; Dipentene; dl-Limonene; Cajeputene; Kautschin\ncpd:C06099\t(R)-Limonene; (+)-Limonene; d-Limonene; (4R)-1-Methyl-4-(prop-1-en-2-yl)cyclohexene; (+)-(R)-Limonene; (+)-(4R)-Limonene\ncpd:C07271\tLimonene-1,2-epoxide; Limonene oxide; (4R)-Limonene-1,2-epoxide\ncpd:C07276\tLimonene-1,2-diol; (1S,2S,4R)-Limonene-1,2-diol; (1S,2S,4R)-Menth-8-ene-1,2-diol\ncpd:C11937\t(1S,4R)-1-Hydroxy-2-oxolimonene; (1S,4R)-1-Hydroxymenth-8-en-2-one\ncpd:C19081\t(4S)-Limonene-1,2-epoxide\ncpd:C19082\t(1R,2R,4S)-Limonene-1,2-diol; (1R,2R,4S)-Menth-8-ene-1,2-diol\ncpd:C19083\t(1R,4S)-1-Hydroxy-2-oxolimonene\n"""  # pylint: disable=line-too-long


def parse_kegg_list(input_str: str) -> dict[str, list[str]]:
    """Parse un résultat du webservice WSDL de KEGG

    Parameters
    ----------
    input_str : str
        le résultat brut txt/raw

    Returns
    -------
    dict[str, list[str]]
        le dictionnaire du résultat : pour chaque clef, la liste des noms

    Examples
    --------
    >>> len(parse_kegg_list(SAMPLE_KEGG_RESULT))
    9
    >>> isinstance(parse_kegg_list(SAMPLE_KEGG_RESULT), dict)
    True

    """
    # http://rest.kegg.jp/find/compound/%22limonene%22
    # result is a list of lines "cpd:id\tname1; name2; ..."
    lines = csv.reader(input_str.split("\n"), delimiter="\t")
    # après split, la dernière ligne est une ligne vide, on évite les vide
    results_dict = {line[0]: line[1].split("; ") for line in lines if line}
    return results_dict


if __name__ == "__main__":
    doctest.testmod(verbose=False, report=True)
