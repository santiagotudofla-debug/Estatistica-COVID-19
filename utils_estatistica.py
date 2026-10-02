import numpy as np

def executar_bootstrap_confirmados(df_filtrado, n_iteracoes, semente):
    rng = np.random.default_rng(int(semente))
    casos_confirmados = df_filtrado["Confirmados"].values
    n = len(casos_confirmados)

    # Reamostragem vetorizada
    amostras = rng.choice(casos_confirmados, size=(int(n_iteracoes), n), replace=True)
    medias_boot = amostras.mean(axis=1)

    media_original = np.mean(casos_confirmados)
    media_bootstrap = np.mean(medias_boot)
    ic_inferior = np.percentile(medias_boot, 2.5)
    ic_superior = np.percentile(medias_boot, 97.5)

    return {
        "n_iteracoes": int(n_iteracoes),
        "media_original": media_original,
        "media_bootstrap": media_bootstrap,
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "medias_boot": medias_boot,
    }
