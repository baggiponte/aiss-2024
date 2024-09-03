from __future__ import annotations

import os
import shutil
import warnings
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import umap
from IPython.display import display
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def load_animals_data(destination):
    """Carica il dataset di foto di animali nella cartella di destinazione"""

    # Nome della cartella in cui verrà estratto il dataset ZIP
    import_folder = "animal-image-dataset-90-different-animals"
    # Percorso del file ZIP contenente il dataset
    path = "/content/animal-image-dataset-90-different-animals.zip"

    # Controlla se la cartella di importazione esiste già
    if not os.path.exists(import_folder):
        # Se non esiste, crea la cartella
        os.makedirs(import_folder)

    # Se la cartella di importazione esiste, rimuovila (pulizia di vecchi dati)
    if os.path.exists(import_folder):
        shutil.rmtree(import_folder)

    # Estrai il contenuto del file ZIP nella cartella di importazione
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(import_folder)

    # Controlla se la cartella di destinazione esiste già
    if os.path.exists(destination):
        # Se esiste, rimuovila per evitare conflitti (pulizia di vecchi dati)
        shutil.rmtree(destination)

    # Sposta il contenuto estratto (specificamente, la cartella 'animals' all'interno di 'import_folder') alla destinazione finale
    shutil.move(os.path.join(import_folder, "animals", "animals"), destination)

    # Pulisci rimuovendo la cartella temporanea di importazione e il file ZIP
    if os.path.exists(import_folder):
        shutil.rmtree(import_folder)
        os.remove(path)


def get_image_embedding(image_path, processor, model):
    """Trasforma singola immagine in un embedding tramite il modello fornito"""

    # Apri l'immagine specificata dal percorso 'image_path'
    # La funzione Image.open carica l'immagine, mentre .convert("RGB") assicura che l'immagine sia in formato RGB
    image = Image.open(image_path).convert("RGB")

    # Preprocessa l'immagine utilizzando il processore fornito
    inputs = processor(images=image, return_tensors="pt")

    # Disabilita il calcolo dei gradienti per l'inferenza (evita l'uso della memoria per il calcolo dei gradienti)
    with torch.no_grad():
        # Passa i dati preprocessati attraverso il modello per ottenere le caratteristiche dell'immagine
        outputs = model.get_image_features(**inputs)

    # Converti l'output delle feature dell'immagine da un tensore PyTorch a un array NumPy
    return outputs.numpy()


def embed_animals_dataset(files):
    """Applica embedding pre-addestrato alle immagini di animali fornite in input"""

    # Carica il modello pre-addestrato CLIP e il processore associato
    # CLIP (Contrastive Language-Image Pretraining) è un modello di visione e linguaggio
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Applica la funzione get_image_embedding a ciascun file di immagine nella lista 'files', una lista di percorsi di file immagine
    # Per ciascun file, 'get_image_embedding' restituisce un embedding dell'immagine
    embeddings = [get_image_embedding(file, processor, model) for file in files]

    # Combina tutti gli embeddings in una singola matrice 2D
    embeddings = np.vstack(embeddings)

    # Restituisce la matrice di embeddings
    return embeddings


def define_my_zoo(animals, zoo_name, origin):
    """Crea cartella con sotto-set di animali selezionati e rinomina i file"""

    # Controlla se la cartella di destinazione esiste già
    # Se esiste, rimuove la cartella e tutto il suo contenuto
    if os.path.exists(zoo_name):
        shutil.rmtree(zoo_name)

    # Inizializza le liste per memorizzare le informazioni sui file, le etichette e i nomi dei file
    labels = []
    files = []
    names = []

    # Cicla attraverso ogni animale nella lista 'animals'
    for animal in animals:
        # Crea il percorso di destinazione per l'animale nella cartella 'zoo_name'
        destination = os.path.join(zoo_name, animal)

        # Se la cartella di destinazione esiste già, rimuovila
        if os.path.exists(destination):
            shutil.rmtree(destination)

        # Copia la cartella dell'animale dalla posizione 'origin' a 'destination'
        shutil.copytree(os.path.join(origin, animal), destination)

        # Ottieni la lista dei file nella cartella dell'animale
        animal_files = [f for f in os.listdir(destination)]

        # Inizializza una lista per memorizzare i nuovi nomi dei file
        animal_names = []
        # Cicla attraverso ogni file nella cartella dell'animale
        for i, file in enumerate(animal_files):
            # Rinomina i file che hanno estensione '.jpg', aggiungendo l'indice al nome del file
            if file.endswith(".jpg"):
                new_name = f"{animal}_{i}.jpg"
                os.rename(
                    os.path.join(destination, file), os.path.join(destination, new_name)
                )
                animal_names.append(new_name)

        # Raccoglie i file rinominati nella cartella dell'animale
        animal_files = [
            os.path.join(destination, file) for file in os.listdir(destination)
        ]
        N = len(animal_files)
        animal_labels = [
            animal for _ in range(N)
        ]  # Crea una lista di etichette per tutti i file dell'animale

        # Aggiungi i file, le etichette e i nomi alla lista complessiva
        labels += animal_labels
        files += animal_files
        names += animal_names

    # Restituisce le liste di file, etichette e nomi
    return files, labels, names


def update_UMAP(
    n_slider,
    d_slider,
    my_pictures,
    embeddings,
    labels,
    names,
):
    """Aggiorna grafico scatter UMAP tramite slider"""

    # Crea un oggetto UMAP con i parametri specificati dagli slider e lo addestra senza le immagini di test
    # n_neighbors: numero di vicini considerati per la riduzione dimensionale
    # min_dist: distanza minima tra i punti nel nuovo spazio
    # n_components: numero di dimensioni per il risultato della riduzione dimensionale (2D)
    # random_state: seme per la randomizzazione, per garantire riproducibilità
    umap_reducer = umap.UMAP(
        n_neighbors=n_slider, min_dist=d_slider, n_components=2, random_state=42
    )

    # Applica la riduzione dimensionale UMAP sugli embeddings esistenti
    embedding_2d = umap_reducer.fit_transform(embeddings)

    # Crea un DataFrame con i risultati della riduzione dimensionale
    embedded_data = pd.DataFrame(embedding_2d, columns=["UMAP_x", "UMAP_y"])
    embedded_data["label"] = labels  # Aggiunge la colonna delle etichette originali
    embedded_data["names"] = names  # Aggiunge la colonna dei nomi dei punti
    embedded_data["size"] = 10  # Imposta una dimensione di marker di base

    # Se esistono immagini di test, applica la stessa riduzione dimensionale UMAP su di esse
    if len(my_pictures) > 0:
        test_embeddings = embed_animals_dataset(
            my_pictures
        )  # Calcola gli embeddings per le immagini di test
        test_embeddings_2d = umap_reducer.transform(
            test_embeddings
        )  # Riduci la dimensione degli embeddings di test con lo stesso riduttore UMAP
        test_embedded_data = pd.DataFrame(
            test_embeddings_2d, columns=["UMAP_x", "UMAP_y"]
        )  # Crea un DataFrame con i risultati della riduzione dimensionale per le immagini di test
        test_embedded_data["label"] = (
            "new"  # Assegna l'etichetta 'new' alle nuove immagini
        )
        test_embedded_data["names"] = [
            pic[9:] for pic in my_pictures
        ]  # Usa una parte del nome del file come testo
        test_embedded_data["size"] = (
            20  # Imposta una dimensione di marker maggiore per le immagini di test
        )
        embedded_data = pd.concat(
            [embedded_data, test_embedded_data], axis=0
        )  # Combina i dati originali e i dati delle immagini di test

    # Crea una mappa di colori per le etichette uniche
    unique_labels = embedded_data["label"].unique()
    color_map = {label: idx for idx, label in enumerate(unique_labels)}
    colors = embedded_data["label"].map(color_map)  # Mappa le etichette ai colori

    # Crea il grafico scatter usando Plotly
    fig = go.FigureWidget(
        go.Scatter(
            x=embedded_data["UMAP_x"],  # Coordinata x dei punti nel grafico
            y=embedded_data["UMAP_y"],  # Coordinata y dei punti nel grafico
            mode="markers",  # Modalità di visualizzazione dei punti come marcatori
            marker=dict(
                color=colors,  # Colore dei marcatori basato sulle etichette
                colorscale="geyser",  # Scala dei colori per la visualizzazione, ref. https://plotly.com/python/colorscales/
                size=embedded_data["size"],  # Dimensione dei marcatori
            ),
            text=embedded_data["names"],  # Testo del tooltip per ogni punto
        )
    )

    # Mostra il grafico nel notebook
    display(fig)

    # Restituisce l'oggetto della figura per eventuali ulteriori utilizzi
    return fig


def update_KMeans(
    k_slider,
    my_pictures,
    umap_reducer,
    embedding_2d,
    labels,
    names,
):
    """Aggiorna grafico scatter K-Means tramite slider"""

    embedded_data = pd.DataFrame(embedding_2d, columns=["UMAP_x", "UMAP_y"])
    embedded_data["label"] = labels
    embedded_data["names"] = names
    embedded_data["size"] = 10

    # Addestra il clustering senza le immagini di test
    kmeans = KMeans(n_clusters=k_slider, random_state=42)
    clustering = kmeans.fit(embedding_2d)

    if len(my_pictures) > 0:
        test_embeddings = embed_animals_dataset(my_pictures)
        test_embeddings_2d = umap_reducer.transform(test_embeddings)
        test_embedded_data = pd.DataFrame(
            test_embeddings_2d, columns=["UMAP_x", "UMAP_y"]
        )
        test_embedded_data["label"] = "new"
        test_embedded_data["names"] = [pic[9:] for pic in my_pictures]
        test_embedded_data["size"] = 20
        embedded_data = pd.concat([embedded_data, test_embedded_data], axis=0)

    # applica il clustering alle immagini di addestramento e di test
    clusters = clustering.predict(embedded_data[["UMAP_x", "UMAP_y"]])
    embedded_data["cluster"] = clusters
    embedded_data["names"] = [
        embedded_data["names"].iloc[i]
        + "_cluster_"
        + str(embedded_data["cluster"].iloc[i])
        for i in range(len(embedded_data))
    ]

    # definisce centroidi dei cluster per la rappresentazione
    centroids = pd.DataFrame(data=kmeans.cluster_centers_, columns=["UMAP_x", "UMAP_y"])
    centroids["label"] = "center"
    centroids["names"] = ["center_" + str(i) for i in np.unique(clusters)]
    centroids["size"] = 15
    centroids["cluster"] = -1
    embedded_data = pd.concat([embedded_data, centroids], axis=0)

    unique_labels = embedded_data["cluster"].unique()
    color_map = {label: idx for idx, label in enumerate(unique_labels)}
    colors = embedded_data["cluster"].map(color_map)

    fig = go.FigureWidget(
        go.Scatter(
            x=embedded_data["UMAP_x"],
            y=embedded_data["UMAP_y"],
            mode="markers",
            marker=dict(
                color=colors,
                # colorscale='bluered',  # ref. https://plotly.com/python/colorscales/
                colorscale="geyser",
                size=embedded_data["size"],
            ),
            text=embedded_data["names"],
        )
    )

    display(fig)
    return fig
