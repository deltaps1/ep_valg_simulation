from typing import Literal
import pandas as pd
import numpy as np
from random import choice
from tqdm import tqdm

def get_valgforbund():
    forbund = [
        ["A", "F", "Å"],
        ["C", "I"],
        ["V", "M", "B"]
    ]
    return forbund

def tildel_forbund(parti, forbund: list[list] | Literal[False] = False):
    if forbund == False:
        forbund = get_valgforbund()
    for x in forbund:
        if parti in x:
            return "".join(x)
    return parti


def calc_forbunds_stemmer(stemmer_parti):
    forbunds_stemmer = (
        stemmer_parti
        .groupby("forbund")
        .stemmer
        .sum()
    )
    return forbunds_stemmer

def gen_runder(df_input, runder=15):
    df = (
        df_input
        .to_frame()
        .rename(columns={"stemmer": "r1"})
    )
    for i in range(2, runder+1):
        df[f"r{i}"] = df["r1"].div(i)
    df = df.T
    stemmedict = {x: df[x].to_list() for x in df.columns}
    return stemmedict

def _get_rundevinder_helper(stemmedict):
    res = {v[0]: k for k, v in stemmedict.items()}
    max_val = max(res)
    vinder = res[max_val]
    stemmedict[vinder].pop(0)
    return vinder, stemmedict

def simulate(stemmedict, runder=15):
    vindere = []
    for _ in range(1, runder+1):
        v, s = _get_rundevinder_helper(stemmedict) 
        vindere.append(v)
        stemmedict = s
    return vindere


def get_partivinder_internal(forbundvinder, stemmer_parti):
    partivindere = {}
    for vinder in set(forbundvinder):
        numb_of_rounds = forbundvinder.count(vinder)
        df_input = (
            stemmer_parti
            .query(f"forbund == @vinder")
            .set_index("parti")
            .stemmer
        )
        stemmedict = gen_runder(df_input, numb_of_rounds)
        subvinder = simulate(stemmedict, numb_of_rounds)
        partivindere[vinder] = subvinder
    return partivindere


def get_partivinder_total(forbundvinder, partivinder):
    rundevinder = []
    for fv in forbundvinder:
        pv = partivinder[fv].pop(0)
        rundevinder.append(pv)
    return rundevinder


def simulate_all(stemmer_parti):
    forbunds_stemmer = calc_forbunds_stemmer(stemmer_parti)
    runde_base = gen_runder(forbunds_stemmer)
    forbundvinder = simulate(runde_base)
    partivinder = get_partivinder_internal(forbundvinder, stemmer_parti)
    tot_vinder = get_partivinder_total(forbundvinder, partivinder)
    return tot_vinder


def draw_scenarios(p_hat, n, antal_scenarier = 10_000):
    mu = n * p_hat
    sigma = np.sqrt(n * p_hat * (1 - p_hat))
    antal_scenarier = 1000
    scenarier = np.random.normal(mu, sigma, antal_scenarier)
    scenarier_i_procenter = scenarier / n
    return scenarier_i_procenter

def setup_stemmer(create_scenarios = True):
    stemmer = (
        pd.read_csv("stemmer")
        .assign(forbund = lambda df: df.parti.apply(tildel_forbund),
                stemmer = lambda df: df.stemmer / 100)
    )
    if create_scenarios:
        stemmer = stemmer.assign(
            scenarier = lambda df: df.stemmer.apply(
                draw_scenarios, args=[n, 50_000]
            )
        )
        
    return stemmer

def opgør_mandater_i_df(stemmer, simulation_result):
    temp_df = (
        stemmer["parti stemmer forbund".split()].copy()
        .assign(mandater = lambda df: df.parti.apply(
            lambda x: simulation_result.count(x))
        )
        .assign(simnum = x)
    )
    return temp_df

if __name__ == "__main__":
    n = 2085
    stemmer = setup_stemmer()
    dfs_res = []
    for x in tqdm(range(10_000)):
        stemmer = (
            stemmer
            .assign(stemmer = lambda df: df.scenarier.apply(lambda x: choice(x)))
        )
        simulation_result = simulate_all(stemmer)
        temp_df = opgør_mandater_i_df(stemmer, simulation_result)
        dfs_res.append(temp_df)
    dfs = pd.concat(dfs_res)
    dfs.to_excel("Simulation.xlsx")
        
