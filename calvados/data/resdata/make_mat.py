import os
import re
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# from multiego.resources import type_definitions
# from multiego.util import masking
# from multiego import io

import argparse
import multiprocessing
import numpy as np
import pandas as pd
# import parmed as pmd
import time
import tarfile
from scipy.special import logsumexp

# d = {
#     type_definitions.gromos_atp.name[i]: type_definitions.gromos_atp.rc_c12[i]
#     for i in range(len(type_definitions.gromos_atp.name))
# }

COLUMNS = ["mi", "ai", "mj", "aj", "c12dist", "p", "cutoff"]

def read_simple_topology(path):
    """
    Reads a simple topology file and extracts box dimensions, periodic boundary conditions (PBC),
    molecule information, atom names, residue names, and residue numbers.

    Parameters
    ----------
    path : str
        Path to the topology file.

    Returns
    -------
    tuple
        A tuple containing:
        - box (list of list of float): Box dimensions.
        - pbc (str): Periodic boundary condition value.
        - n_mol (list of int): Number of molecules.
        - molecules (list of str): Molecule names.
        - atom_names (list of list of str): Atom names for each molecule.
        - residue_names (list of list of str): Residue names for each molecule.
        - residue_numbers (list of list of int): Residue numbers for each molecule.
    """
    box_found = False
    molecule_found = False
    pbc_found = False
    box = []
    pbc = None
    n_mol = []
    molecules = []
    atom_names = []
    residue_names = []
    residue_numbers = []

    try:
        with open(path, 'r') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    print("Detected empty line. Skipping...")
                    continue

                if "box" in line:
                    if box_found:
                        raise RuntimeError("Multiple box definitions found in the topology file")
                    box_found = True
                    values = line.split()
                    for i in range(1, len(values), 3):
                        box.append([float(values[i]), float(values[i + 1]), float(values[i + 2])])

                elif "molecule" in line:
                    molecule_found = True
                    values = line.split()
                    molecule_name = values[1]
                    molecules.append(molecule_name)
                    atom_names.append([])
                    residue_names.append([])
                    residue_numbers.append([])

                    for value in values[2:]:
                        if value.isdigit():
                            n_mol.append(int(value))
                            break
                        else: # TODO check if this is correct
                            if "_" in value or "-" in value:
                                parts = value.split("_")
                                atom_name = parts[0]
                                residue_name, resnum = parts[1].split("-")
                                residue_names[-1].append(residue_name)
                                atom_names[-1].append(atom_name)
                                residue_numbers[-1].append(int(resnum))
                            else:
                                residue_names[-1].append("XXX")
                                atom_names[-1].append(value)
                                residue_numbers[-1].append(1)

                elif "pbc" in line:
                    if pbc_found:
                        raise RuntimeError("Multiple PBC definitions found in the topology file")
                    pbc_found = True
                    pbc = line.split()[1]
                    print(f"Found PBC value: {pbc}")

        if not box_found:
            raise RuntimeError("No box found in the topology file")
        if not molecule_found:
            raise RuntimeError("No molecule found in the topology file")
        if not pbc_found:
            pbc = "unset"

    except FileNotFoundError:
        raise RuntimeError("Cannot find the indicated topology file")

    # construct the molecules dictionary
    molecules_dict = { k: {} for k in molecules }
    for k in molecules:
        molecules_dict[k]["atoms"] = atom_names[molecules.index(k)]
        molecules_dict[k]["residue"] = residue_names[molecules.index(k)]
        molecules_dict[k]["residue_index"] = residue_numbers[molecules.index(k)]
        molecules_dict[k]["name"] = k

    n_molecule_types = len(molecules)

    return molecules_dict


def write_mat(df, output_file):
    if df.empty:  # Check if the DataFrame is empty
        print(f"Warning: The DataFrame is empty. No file will be written to {output_file}.")
        return

    df = df.rename(
        columns={
            "mi": "molecule_name_ai",
            "ai": "ai",
            "mj": "molecule_name_aj",
            "aj": "aj",
            "c12dist": "distance",
            "p": "probability",
        }
    )

    df["molecule_name_ai"] = df["molecule_name_ai"].astype("category")
    df["ai"] = df["ai"].astype("category")
    df["molecule_name_aj"] = df["molecule_name_aj"].astype("category")
    df["aj"] = df["aj"].astype("category")
    df["distance"] = df["distance"].astype("float64")
    df["probability"] = df["probability"].astype("float64")
    df["cutoff"] = df["cutoff"].astype("float64")
    df["learned"] = True

    # Force the column order
    ordered_columns = ["molecule_name_ai", "ai", "molecule_name_aj", "aj", "distance", "probability", "cutoff"]#, "learned"]
    df = df[ordered_columns]

    df.to_csv(output_file, index=False, sep=" ", header=False)


def read_mat(name, protein_ref_indices, args, cumulative=False):
    path_prefix = f"{args.histo}"
    ref_df = pd.read_csv(f"{path_prefix}/{name}", header=None, sep=r"\s+", usecols=[0, *protein_ref_indices])
    ref_df_columns = ["distance", *[str(x) for x in protein_ref_indices]]
    ref_df.columns = ref_df_columns
    ref_df.set_index("distance", inplace=True)

    return ref_df


def zero_probability_decorator(func, flag):
    """
    Decorator of function to return 0 if flag is rased
    """

    def wrapper(*args, **kwargs):
        if flag:
            return 0  # Return 0 if the flag is set
        return func(*args, **kwargs)  # Otherwise, execute the original function

    return wrapper


def run_mat_(arguments):
    """
    Preforms the main routine of the histogram analysis to obtain the intra- and intermat files.
    Is used in combination with multiprocessing to speed up the calculations.

    Parameters
    ----------
    arguments : dict
        Contains all the command-line parsed arguments

    Returns
    -------
    out_path : str
        Path to the temporary file which contains a partial pd.DataFrame with the analyzed data
    """
    (
        args,
        protein_ref_indices_i,
        protein_ref_indices_j,
        original_size_j,
        c12_cutoff,
        mi,
        mj,
        frac_target_list,
        mat_type,
    ) = arguments
    process = multiprocessing.current_process()
    df = pd.DataFrame(columns=COLUMNS)
    # We do not consider old histograms
    frac_target_list = [x for x in frac_target_list if x[0] != "#" and x[-1] != "#"]
    for i, ref_f in enumerate(frac_target_list):
        print(f"\rProgress: {ref_f} ", end="", flush=True)
        results_df = pd.DataFrame()
        ai = ref_f.split(".")[-2].split("_")[-1]

        all_ai = [ai for _ in range(1, original_size_j + 1)]
        range_list = [str(x) for x in range(1, original_size_j + 1)]

        results_df["ai"] = np.array(all_ai).astype(int)
        results_df["mi"] = mi
        results_df["aj"] = np.array(range_list).astype(int)
        results_df["mj"] = mj
        results_df["c12dist"] = 0.0
        results_df["p"] = 0.0
        results_df["cutoff"] = 0.0

        if np.isin(int(ai), protein_ref_indices_i):
            cut_i = np.where(protein_ref_indices_i == int(ai))[0][0]

            # column mapping
            ref_df = read_mat(ref_f, protein_ref_indices_j, args)
            ref_df.loc[len(ref_df)] = c12_cutoff[cut_i]

            c12dist = ref_df.apply(lambda x: c12_avg(ref_df.index.to_numpy(), weights=x.to_numpy()), axis=0).values
            if mat_type == "intra":
                p = ref_df.apply(
                    lambda x: calculate_probability(ref_df.index.to_numpy(), weights=x.to_numpy()),
                    axis=0,
                ).values
            elif mat_type == "inter":
                # repeat for cumulative
                c_ref_f = ref_f.replace("inter_mol_", "inter_mol_c_")
                c_ref_df = read_mat(c_ref_f, protein_ref_indices_j, args, True)
                c_ref_df.loc[len(c_ref_df)] = c12_cutoff[cut_i]
                p = c_ref_df.apply(
                    lambda x: get_cumulative_probability(c_ref_df.index.to_numpy(), weights=x.to_numpy()),
                    axis=0,
                ).values

            results_df.loc[results_df["aj"].isin(protein_ref_indices_j), "c12dist"] = c12dist
            results_df.loc[results_df["aj"].isin(protein_ref_indices_j), "p"] = p
            results_df.loc[results_df["aj"].isin(protein_ref_indices_j), "cutoff"] = c12_cutoff[cut_i].astype(float)

        if df.empty:
            df = results_df.copy()
        else:
            if not results_df.empty:
                df = pd.concat([df, results_df])

    print("done.")
    df.fillna(0).infer_objects(copy=False)
    out_path = f"mat_{process.pid}_t{time.time()}.part"
    df.to_csv(out_path, index=False)

    return out_path


def get_col_params(values, weights):
    """
    TODO rename pls

    Preprocesses arrays (histograms) to allow for proper analysis. Last values are removed from the arrays
    and should correspond to the respective cutoff for the histogram. The histograms are truncated
    according to the cutoff.

    Parameters
    ----------
    values : np.array
        The array of the histograms x values
    weights : np.array
        The array with the respective weights

    Returns
    -------
    cutoff : float
        The cutoff which is deduced by reading the last value of the weights array
    i : int
        The index at which the cutoff is greter or equal than the values array
    norm : float
        The new normalization constant after truncation
    v : np.array
        The truncated x values of the histogram according to the cutoff
    w : np.array
        The truncated weights of the histogram according to the cutoff
    """
    v = values[:-1]
    cutoff = weights[len(weights) - 1]
    w = weights[:-1]
    i = np.where(v <= cutoff)
    if not np.any(i):
        return 0, 0, 0, 0, 0  # check if empty
    i = i[0]
    w = w[i]
    v = v[i]
    norm = np.sum(w)
    i = i[-1]
    return cutoff, i, norm, v, w


def calculate_probability(values, weights):
    """
    Calculates a plain probability accoring to sum_x x * dx

    Parameters
    ----------
    values : np.array
        The array of the histograms x values
    weights : np.array
        The array with the respective weights

    Returns
    -------
    The probability of the histogram
    """
    dx = values[1] - values[0]
    cutoff, i, norm, v, w = get_col_params(values, weights)
    return np.minimum(np.sum(w * dx), 1)


def get_cumulative_probability(values, weights):
    cutoff, i, norm, v, w = get_col_params(values, weights)
    return weights[i]


def c12_avg(values, weights):
    """
    Calculates the c12 exp averaging of a histogram

    Parameters
    ----------
    values : np.array
        The array of the histograms x values
    weights : np.array
        The array with the respective weights

    Returns
    -------
    The c12 average
    """
    cutoff, i, norm, v, w = get_col_params(values, weights)
    if np.sum(w) == 0:
        return 0
    r = np.where(w > 0.0)
    # fmt: off
    v = v[r[0][0]:v.size]
    w = w[r[0][0]:w.size]
    # fmt: on
    res = np.maximum(cutoff / 4.5, 0.1)
    log_exp_sum = logsumexp(1.0 / v / res, b=w) - np.log(norm)
    exp_aver = (1.0 / res) / log_exp_sum
    if exp_aver < 0.01:
        exp_aver = 0

    return exp_aver


def warning_cutoff_histo(cutoff, max_adaptive_cutoff):
    """
    Prints warning if the histogram cutoff is smaller as the maximum adaptive cutoff.

    Parameters
    ----------
    cutoff : float
        The cutoff of the histogram calculations. Parsed from the command-line in the standard programm.
    max_adaptive_cutoff : float
        The maximum adaptive cutoff calculated from the LJ c12 parameters.
    """
    print(
        f"""
    #############################

    -------------------
    WARNING
    -------------------

    Found an adaptive cutoff greater then the cutoff used to generate the histogram:
    histogram cutoff = {cutoff}
    maximum adaptive cutoff = {max_adaptive_cutoff}

    Be careful!. This could create errors.
    If this is not wanted, please recalculate the histograms setting the cutoff to at least cutoff={max_adaptive_cutoff}

    #############################
    """
    )


def calculate_matrices(args):
    """
    Starts the main routine for calculating the intermat by:
     - reading the topologies
     - figuring out all the interacting molecules
     - calculating the cutoffs
     - and caclulating the probabilities
    The operation is finalized by writing out a csv with the name pattern intermat<_name>_{mol_i}_{mol_j}.ndx

    Parameters
    ----------
    args : dict
        The command-line parsed parameters
    """
    # topology_mego, topology_ref, N_species, molecules_name, mol_list = read_topologies(args.mego_top, args.target_top)
    # topology_mego, topology_ref = read_topologies(args.mego_top, args.target_top)
    topology_mego = read_simple_topology(args.mego_top)
    topology_ref = read_simple_topology(args.target_top)
    n_species = len(topology_mego.keys())
    molecules_name = list(topology_mego.keys())
    mol_list = np.arange(1, n_species + 1, 1)

    print(
        f"""
    Topology contains {n_species} molecules species. Namely {molecules_name}.
    Calculating intermat for all species\n\n
    """
    )
    for mol_i in mol_list:
        if args.intra:
            prefix = f"intra_mol_{mol_i}_{mol_i}"
            main_routine(mol_i, mol_i, topology_mego, topology_ref, molecules_name, args.calvados_ff, prefix)
        # fmt: off
        for mol_j in mol_list[mol_i - 1:]:
            # fmt: on
            if mol_i == mol_j and not args.same:
                continue
            if mol_i != mol_j and not args.cross:
                continue

            prefix = f"inter_mol_{mol_i}_{mol_j}"
            main_routine(mol_i, mol_j, topology_mego, topology_ref, molecules_name, args.calvados_ff, prefix)


def main_routine(mol_i, mol_j, topology_mego, topology_ref, molecules_name, ff, prefix):
    # TODO introduce assert check to see all molecule names are equal
    df = pd.DataFrame(columns=COLUMNS)
    # define matrix type (intra o inter)
    mat_type = prefix.split("_")[0]
    print(
        f"\nCalculating {mat_type} between molecule {mol_i} and {mol_j}: {molecules_name[mol_i-1]} and {molecules_name[mol_j-1]}"
    )
    target_list = [x for x in os.listdir(args.histo) if prefix in x and x.endswith(".dat")]

    mol_name_i = molecules_name[mol_i - 1]
    mol_name_j = molecules_name[mol_j - 1]
    protein_ref_i = topology_ref[mol_name_i]
    protein_ref_j = topology_ref[mol_name_j]
    original_size_j = len(set(protein_ref_j['residue_index']))

    residue_list_i, residue_index_i = zip(*list(dict.fromkeys(zip(protein_ref_i['residue'], protein_ref_i['residue_index']))))
    residue_list_j, residue_index_j = zip(*list(dict.fromkeys(zip(protein_ref_j['residue'], protein_ref_j['residue_index']))))
    residue_list_i = np.array(residue_list_i)
    residue_list_j = np.array(residue_list_j)
    residue_index_i = np.array(residue_index_i)
    residue_index_j = np.array(residue_index_j)

    EPSILON = 0.8368 # kJ/mol
    type_sigma_dict = {k: v for k, v in ff[['three', 'sigmas']].values}
    # c6_dict = { k: 4 * EPSILON * v ** 6 for k, v in ff[['three', 'sigmas']].values }
    # c12_dict = { k: 4 * EPSILON * v ** 12 for k, v  in ff[['three', 'sigmas']].values }

    # Good-Hope combination rule
    # calculate_cutoff = lambda x: np.argmin((np.cumsum(x[x > 0.0]) / np.sum(x[x > 0.0])) > 0.95)
    # lj_map_gen = lambda c6, c12: lambda x: c12 / x ** 12 - c6 / x ** 6

    combined_sigma = np.sqrt(
        np.vectorize(type_sigma_dict.get)(residue_list_i) * np.vectorize(type_sigma_dict.get)(residue_list_j)[:, np.newaxis]
    )
    combined_c6 = 4 * EPSILON * combined_sigma ** 6
    combined_c12 = 4 * EPSILON * combined_sigma ** 12
    np.savetxt("combined_sigma.txt", combined_sigma, fmt="%s")
    # lj_map = lj_map_gen(combined_c6, combined_c12) # function of r with c6 and c12 in place
    r = np.linspace(0, args.cutoff, 100)
    # lj_potentials = np.array([ lj_map(r_i) for r_i in r ])
    # print(lj_potentials[0])
    # cutoff = calculate_cutoff(lj_potentials)

    r = np.linspace(0, args.cutoff, 100)
    # SLOW VERSION
    cutoff = np.zeros((len(residue_list_i), len(residue_list_j)))
    for i in range(len(residue_list_i)):
        for j in range(len(residue_list_j)):
            c6, c12 = combined_c6[i, j], combined_c12[i, j]
            lj = c12 / r ** 12 - c6 / r ** 6
            attractive_mask = lj < 0
            att_lj = np.where(attractive_mask, lj, 0)
            att = np.where(attractive_mask, np.cumsum(att_lj), 0)
            total_att = np.sum(att_lj)
            idx = np.argwhere((att / total_att) > 0.95)
            cutoff[i, j] = r[idx[0][0]]
    # FAST VERSION
    # cc6, cc12 = combined_c6[..., np.newaxis], combined_c12[..., np.newaxis]
    # lj = cc12 / r[np.newaxis, np.newaxis, :] ** 12 - cc6 / r[np.newaxis, np.newaxis, :] ** 6
    # attractive_mask = lj < 0
    # att_lj = np.where(attractive_mask, lj, 0)
    # att = np.where(attractive_mask, np.cumsum(att_lj, axis=2), 0)
    # total_att = np.sum(att_lj, axis=2)
    # print(total_att[..., np.newaxis].shape)
    # idx = np.argwhere((att / total_att[..., np.newaxis]) > 0.95)
    # # print(idx.shape)
    # att_percent = att / total_att[..., np.newaxis]
    # print(att_percent.shape)
    # idx = att_percent > 0.95
    # a = np.argwhere(idx)
    # print(a.shape)
    # # print(idx)
    # # print(f"idx {list(idx)}")
    # cutoff = r[idx[:, 0], idx[:, 1], idx[:, 2]]

    if np.any(cutoff > args.cutoff):
        warning_cutoff_histo(args.cutoff, np.max(cutoff))
    if np.isnan(cutoff.astype(float)).any():
        warning_cutoff_histo(args.cutoff, np.max(cutoff))

    if args.zero:
        df = pd.DataFrame()
        df["ai"] = np.repeat(len(residue_list_i), len(residue_list_j))
        df["mi"] = [mol_i for _ in range(len(residue_list_i) * len(residue_list_j))]
        df["aj"] = np.tile(len(residue_list_j), len(residue_list_i))
        df["mj"] = [mol_j for _ in range(len(residue_list_i) * len(residue_list_j))]
        df["c12dist"] = 0.0
        df["p"] = 0.0
        df["cutoff"] = cutoff.flatten()
    else:
        chunks = np.array_split(target_list, args.num_threads)
        ##########################
        # PARALLEL PROCESS START #
        ##########################

        pool = multiprocessing.Pool(args.num_threads)
        results = pool.map(
            run_mat_,
            [
                (
                    args,
                    residue_index_i,
                    residue_index_j,
                    original_size_j,
                    cutoff,
                    mol_i,
                    mol_j,
                    x,
                    mat_type,
                )
                for x in chunks
            ],
        )
        pool.close()
        pool.join()

        ########################
        # PARALLEL PROCESS END #
        ########################

        # concatenate and remove partial dataframes
        for name in results:
            try:
                part_df = pd.read_csv(name)
                df = pd.concat([df, part_df])
            except pd.errors.EmptyDataError:
                print(f"Ignoring partial dataframe in {name} as csv is empty")
        [os.remove(name) for name in results]
        df = df.astype({"mi": "int32", "mj": "int32", "ai": "int32", "aj": "int32"})

        df = df.sort_values(by=["mi", "mj", "ai", "aj"])
        df.drop_duplicates(subset=["mi", "ai", "mj", "aj"], inplace=True)

    df["mi"] = df["mi"].map("{:}".format)
    df["mj"] = df["mj"].map("{:}".format)
    df["ai"] = df["ai"].map("{:}".format)
    df["aj"] = df["aj"].map("{:}".format)
    df["c12dist"] = df["c12dist"].map("{:,.6f}".format)
    df["p"] = df["p"].map("{:,.6e}".format)
    df["cutoff"] = df["cutoff"].map("{:,.6f}".format)

    df.index = range(len(df.index))
    out_name = args.out_name + "_" if args.out_name else ""
    output_file = f"{args.out}/{mat_type}mat_{out_name}{mol_i}_{mol_j}.ndx"
    print(f"Saving output for molecule {mol_i} and {mol_j} in {output_file}")
    write_mat(df, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--histo",
        type=str,
        required=False,
        help='Path to the directory containing the histograms. The histogram files should contain the prefix "intra_" for intra molecular contact descriptions and "inter_" for  inter molecular.',
    )
    parser.add_argument(
        "--target_top",
        required=True,
        help="Path to the topology file of the system on which the histograms were calculated on",
    )
    parser.add_argument(
        "--mego_top",
        required=True,
        help="""Path to the standard multi-eGO topology of the system generated by pdb2gmx""",
    )
    parser.add_argument(
        "--mode", help="Sets the caculation to be intra/same/cross for histograms processing", default="intra+same+cross"
    )
    parser.add_argument("--bkbn_H", help="Name of backbone hydrogen (default H, charmm HN)", default="H")
    parser.add_argument("--out", default="./", help="""Sets the output path""")
    parser.add_argument(
        "--out_name",
        help="""Sets the output name of files to be added to the default one: intermat_<out_name>_mi_mj.ndx or intramat_<out_name>_mi_mj.ndx""",
    )
    parser.add_argument(
        "--num_threads",
        default=1,
        type=int,
        help="Sets the number of processes to perform the calculation",
    )
    parser.add_argument(
        "--cutoff",
        default=0.75,
        type=float,
        help="To be set to the max cutoff used for the accumulation of the histograms",
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help="Read from tar file instead of directory",
    )
    parser.add_argument(
        "--custom_c12",
        type=str,
        help="Custom dictionary of c12 for special molecules",
    )
    parser.add_argument(
        "--zero",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--calvados_ff",
        type=str,
        help="Force field to be used for calculations",
        required=True,
    )
    args = parser.parse_args()

    # check either histo or zero flag are set
    if not args.histo and not args.zero:
        raise ValueError("Either --histo or --zero flag must be set.")
    if args.histo and args.zero:
        raise ValueError("Both --histo and --zero flags cannot be set at the same time.")

    # check if output file exists
    if not os.path.exists(args.out):
        print(f"The path '{args.out}' does not exist.")
        sys.exit()

    if not args.zero and not args.tar:
        if not os.path.isdir(args.histo):
            print(f"The path '{args.histo}' is not a directory.")
            sys.exit()

    if not args.zero and args.tar:
        if not tarfile.is_tarfile(args.histo):
            print(f"The path '{args.histo}' is not a tar file.")
            sys.exit()

    args.calvados_ff = pd.read_csv(args.calvados_ff, sep=",")

    # Sets mode
    modes = np.array(args.mode.split("+"), dtype=str)
    modes_possible = np.array(["intra", "same", "cross"])
    args.intra = False
    args.same = False
    args.cross = False

    if not np.any(np.isin(modes, modes_possible)):
        raise ValueError(
            f"inserted mode {args.mode} is not correct and got evaluated to {modes}. Choose intra,same and or cross separated by '+', e.g.: intra+same or same+cross"
        )

    if "intra" in modes:
        args.intra = True
    if "same" in modes:
        args.same = True
    if "cross" in modes:
        args.cross = True

    N_BINS = args.cutoff / (0.01 / 4)
    DX = args.cutoff / N_BINS
    CUTOFF_FACTOR = 1.45
    print(
        f"""
    Starting with cutoff = {args.cutoff},
                  n_bins = {N_BINS},
                  dx     = {DX}
                  on {args.num_threads} threads
    """
    )

    calculate_matrices(args)
