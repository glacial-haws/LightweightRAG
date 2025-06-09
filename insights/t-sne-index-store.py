import logging
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

import cbfk.experiment_config
import cbfk.index_manager
from cbfk.log.log_config import LogConfig as LC
from insights.plot_utils import PlotUtils as PU

LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_embeddings_and_labels(index) -> tuple[np.ndarray, list[str]]:
    """
    Extract embeddings and assign category labels based on file_name.
    Categories: Master Service Agreement, Solution, Schedule, Annex, Amendment.
    First category found in file_name is used as label. If none match, label is 'Other'.
    """
    import numpy as np

    vector_store = index.vector_store
    data = vector_store._data

    # Try common attribute names
    if hasattr(data, "id_list") and hasattr(data, "embedding_list"):
        node_ids = data.id_list
        embeddings = data.embedding_list
    elif hasattr(data, "ids") and hasattr(data, "embeddings"):
        node_ids = data.ids
        embeddings = data.embeddings
    elif hasattr(data, "embedding_dict"):
        node_ids = list(data.embedding_dict.keys())
        embeddings = list(data.embedding_dict.values())
    else:
        raise AttributeError(f"Cannot find node id and embedding attributes on {type(data)}. Available: {dir(data)}")

    docstore = index.storage_context.docstore.docs
    categories = [
        "Master Service Agreement",
        "Addendum",
        "Amendment",
        "Schedule",
        "Annex",
    ]

    def label_from_filename(file_name: str) -> str:
        file_name_lower = file_name.lower()
        for category in categories:
            if category.lower() in file_name_lower:
                return category
        return "Other"

    labels = [
        label_from_filename(docstore[node_id].metadata.get("file_name", "Unknown"))
        for node_id in node_ids
    ]
    return np.array(embeddings), labels



def plot_tsne(embeddings: np.ndarray, labels: list[str], title: str, ax) -> tuple[list, list[str]]:
    from collections import Counter
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    # Use N distinct shades from Oranges colormap
    N = len(set(labels))
    cmap = plt.get_cmap('Oranges')
    colors = cmap(np.linspace(0.2, 0.9, N))  # Avoid too light/dark extremes
    ax.scatter(reduced[:, 0], reduced[:, 1], c=[colors[i] for i in encoded_labels], alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # Count occurrences for each label
    label_counts = Counter(labels)
    # Prepare tuples for sorting: (count, handle, label)
    label_names = le.classes_
    cmap = plt.get_cmap('Oranges')
    N = len(label_names)
    legend_colors = cmap(np.linspace(0.2, 0.9, N))
    label_info = []
    for i, label in enumerate(label_names):
        count = label_counts[label]
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=legend_colors[i], markersize=8, label=label)
        legend_label = f"{label} ({count})"
        label_info.append((count, handle, legend_label))
    # Sort by ascending count (least frequent first)
    label_info.sort(key=lambda x: x[0])
    # Assign darkest color to least frequent, lightest to most frequent
    color_indices = np.linspace(0.9, 0.2, len(label_info))  # Darkest to lightest
    label_to_color = {}
    handles = []
    legend_labels = []
    for idx, (_, _, legend_label) in enumerate(label_info):
        # Extract label name from legend_label (which is 'Label (count)')
        label = legend_label.rsplit(' (', 1)[0]
        color = cmap(color_indices[idx])
        label_to_color[label] = color
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=legend_label)
        handles.append(handle)
        legend_labels.append(legend_label)
    # Assign scatter colors by label
    scatter_colors = [label_to_color[label] for label in labels]
    ax.scatter(reduced[:, 0], reduced[:, 1], c=scatter_colors, alpha=0.7)
    return handles, legend_labels




def load_index_from_dir(persist_dir: Path, embedding_model: str):
    experiment = cbfk.experiment_config.ExperimentConfig(
                                llm_model = "no llm", 
                                llm_temperature = 0.0,
                                llm_max_tokens = 2000,
                                similarity_top_k = 10,
                                bm25_top_k = 0,
                                vector_top_k = 10,
                                rag_prompt = "no prompt",
                                embedding_model = embedding_model,
                                splitter_type = cbfk.experiment_config.SplitterType.SENTENCE,
                                chunk_size = 512,
                                chunk_overlap_pct = 0.1,   
                                )
    start_load = time.time()
    index, _, _ = cbfk.index_manager.load_index(persist_dir, experiment, True)
    logger.info(f"Loaded index from {persist_dir} in {time.time() - start_load:.2f} seconds")
    return index




def get_char_differences(folders: list[str], reference: str) -> str:
    """
    Returns the segment(s) (split by '_') in 'reference' that are different from the other folder names in 'folders'.
    For example, for folders = ['A_token', 'A_sentence', 'A_semantic'] and reference = 'A_token', returns 'token'.
    """
    if reference not in folders:
        raise ValueError("Reference folder must be in the folders list.")
    if len(folders) < 2:
        return ""
    ref_parts = reference.split('_')
    other_parts = [f.split('_') for f in folders if f != reference]
    unique_segments: list[str] = []
    for i, ref_seg in enumerate(ref_parts):
        segs_at_i = [parts[i] if i < len(parts) else None for parts in other_parts]
        if not all(seg == ref_seg for seg in segs_at_i):
            unique_segments.append(ref_seg)
    return '_'.join(unique_segments)



@dataclass
class FolderInfo:
    folder: str
    model: str
    splitter: str
    chunks: str
    title: str = None   


def parse_folder_names(folders: list[Path]) -> list[FolderInfo]:
    result = []
    for folder in folders:
        splits = folder.name.split('_')
        if len(splits) != 4:
            raise ValueError(f"Unexpected directory name: {folder}")
        fi = FolderInfo(folder, splits[1], splits[2], splits[3])
        fi.title = get_char_differences([f.name for f in folders], folder.name)
        result.append(fi)
    return result



def main():

    args = PU.parse_args()
    PU.init_plot_style(args.light)

    if len(args.folders) == 0:  
        raise ValueError("No folders specified")
    
    index_info = parse_folder_names(args.folders)

    n = len(args.folders)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Prepare for a global legend
    legend_handles = None
    legend_labels = None
    for i, idx_info in enumerate(index_info):
        row, col = divmod(i, cols)
        ax = axes[row][col]
        persist_dir = Path(idx_info.folder)

        index = load_index_from_dir(persist_dir, idx_info.model)
        embeddings, labels = extract_embeddings_and_labels(index)
        
        if len(embeddings) > 0:
            handles, label_names = plot_tsne(embeddings, labels, title=idx_info.title, ax=ax)
            # Save legend handles/labels from the first non-empty plot
            if legend_handles is None:
                legend_handles = handles
                legend_labels = label_names
        else:
            ax.set_title(f"{idx_info.folder.name} (no data)")
            ax.axis('off')

    # Plot a single legend above the row, below the subtitle if any plot has data
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(legend_labels),
            frameon=False,
            fontsize='medium',
        )

    plt.tight_layout()
    # Save the plot in the current directory with a common name of all folders plotted
    from os.path import commonprefix
    folder_names = [Path(f).name for f in args.folders]
    common = commonprefix(folder_names)
    if not common or common == "_":
        common = "_".join(folder_names)
    filename = Path('results') / f"{common}_tsne.png"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved t-SNE plot as {filename}")
    #plt.show()


if __name__ == "__main__":
    main()
