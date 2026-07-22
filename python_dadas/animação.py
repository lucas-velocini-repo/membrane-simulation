import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animar_corda_from_csv(
    csv_path,
    L,
    N_modos,
    fs,
    num_frames=200,
    x_pontos=100,
    y_lim=(-1.2, 1.2),
    salvar_video=False,
    nome_video='corda_vibrando.mp4'
):
    """
    Gera animação da corda a partir do CSV de respostas modais.

    Parâmetros:
        csv_path (str): caminho do CSV.
        L (float): comprimento da corda (m).
        N_modos (int): número de modos.
        fs (float): taxa de amostragem.
        num_frames (int): número de frames da animação.
        x_pontos (int): resolução espacial (número de pontos ao longo da corda).
        y_lim (tuple): limites do eixo y.
        salvar_video (bool): se True, salva como MP4.
        nome_video (str): nome do arquivo de vídeo.
    """
    # Carregar CSV
    df = pd.read_csv(csv_path)
    tempos = df['tempo (s)'].values
    total_amostras = len(tempos)

    # Extrair modos
    modos = []
    for n in range(1, N_modos+1):
        col_name = f'modo_{n}'
        if col_name not in df.columns:
            raise ValueError(f"Coluna {col_name} não encontrada.")
        modos.append(df[col_name].values)
    respostas = np.array(modos)  # (N_modos, amostras)

    # Subamostragem para os frames
    step = max(1, total_amostras // num_frames)
    indices = np.arange(0, total_amostras, step)[:num_frames]
    tempos_frame = tempos[indices]

    # Malha espacial
    x_espaco = np.linspace(0, L, x_pontos)

    # Pré-calcular todos os frames (opcional, mas mais rápido)
    U_all = np.zeros((len(indices), x_pontos))
    for i, idx in enumerate(indices):
        u = np.zeros(x_pontos)
        for n in range(N_modos):
            u += np.sin((n+1) * np.pi * x_espaco / L) * respostas[n, idx]
        # Normalizar cada frame para que a amplitude fique visível
        max_abs = np.max(np.abs(u))
        if max_abs > 1e-12:
            u = u / max_abs
        U_all[i, :] = u

    # Configurar figura
    fig, ax = plt.subplots(figsize=(10,4))
    line, = ax.plot(x_espaco, U_all[0, :], lw=2)
    ax.set_xlim(0, L)
    ax.set_ylim(y_lim)
    ax.set_xlabel('Posição x (m)')
    ax.set_ylabel('Deslocamento (normalizado)')
    ax.grid(True)

    def animate(i):
        line.set_ydata(U_all[i, :])
        ax.set_title(f't = {tempos_frame[i]:.3f} s')
        return line,

    ani = FuncAnimation(fig, animate, frames=len(indices), interval=50, blit=True)

    if salvar_video:
        # Requer ffmpeg instalado
        ani.save(nome_video, writer='ffmpeg', fps=20)
        print(f"Vídeo salvo como '{nome_video}'")
    else:
        plt.show()

# Chame a função com os parâmetros que você usou na simulação
animar_corda_from_csv(
    csv_path='respostas_modais_subamostradas.csv',
    L=1.0,
    N_modos=20,
    fs=44100,          # substitua pelo valor real
    num_frames=300,    # quantos quadros na animação
    x_pontos=100,
    y_lim=(-1.2, 1.2),
    salvar_video=True,   # ou False para exibir janela
    nome_video='corda_vibrando.mp4'
)