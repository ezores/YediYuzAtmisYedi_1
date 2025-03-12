
from pyvis.network import Network
import webbrowser

def generate_network_visualization(mlp, output_file="Visualization_MLP.html", max_neurons=20):
    """
    Visualisation interactive d'un MLP avec PyVis:
    Ce script génère une visualisation interactive d'un perceptron multicouche (MLP) sous forme de réseau.
    
    Les étapes de la visualisation :
    - Les neurones sont de forme 'dot' (taille = 50) et sont disposés en colonnes (couches).
    - Les neurones de la couche d'entrée n'affichent pas de label ; ceux des couches cachées et de sortie
      affichent leur biais sous la forme (b=XX.XX).
    - Les liens (edges) sont visibles dès le départ, mais sans label de poids.
    - Lorsque l'utilisateur clique sur un lien, la valeur du poids du lien s'affiche.
    - Le placement des neurones et des liens est manuel, aucune physique n'est activée.

    Arguments:
        mlp : MLP
            Un modèle perceptron multicouche (MLP) dont la structure sera visualisée.
        output_file : str
            Nom du fichier HTML de sortie contenant la visualisation. Par défaut "Visualization_MLP.html".
        max_neurons : int
            Limite du nombre de neurones à afficher par couche (par défaut 20).
    """
    
    # Création du réseau interactif avec PyVis
    net = Network(height="800px", width="100%", directed=True)

    # 1) Déterminer la taille de chaque couche
    layer_sizes = [mlp.input_size] + mlp.hidden_sizes + [mlp.output_size]
    num_layers = len(layer_sizes)

    # 2) Limiter chaque couche à 'max_neurons'
    sampled_indices = []
    for size in layer_sizes:
        sampled_indices.append(list(range(min(size, max_neurons))))

    # 3) Définir des couleurs pour chaque couche
    layer_colors = ["#ADD8E6", "#90EE90", "#FFB6C1", "#FFA500", "#DDA0DD", "#87CEFA", "#FFC0CB"]

    # 4) Paramètres de positionnement manuel des neurones
    x_spacing = 300
    y_spacing = 100
    title_margin = 70
    bottom_margin = 60

    # 5) Créer les neurones (nodes) pour chaque couche
    for layer_idx, neuron_indices in enumerate(sampled_indices):
        color = layer_colors[layer_idx % len(layer_colors)]
        n_neurons = len(neuron_indices)
        y_offset = (n_neurons - 1) * y_spacing / 2

        for i, neuron_idx in enumerate(neuron_indices):
            node_id = f"node_{layer_idx}_{neuron_idx}"
            label = " "  # Pas de label pour la couche d'entrée

            # Affichage des biais pour les couches cachées + sortie
            if layer_idx > 0 and hasattr(mlp, "biases"):
                try:
                    bias_val = mlp.biases[layer_idx - 1][neuron_idx, 0]
                    label = f"(b={bias_val:.2f})"
                except Exception:
                    pass

            # Calcul de la position x et y pour chaque neurone
            x_coord = layer_idx * x_spacing
            y_coord = i * y_spacing - y_offset

            # Ajouter le neurone au réseau
            net.add_node(
                node_id,
                label=label,
                title=label,
                x=x_coord,
                y=y_coord,
                color=color,
                shape="dot",
                size=50,
                font={
                    "color": "black",
                    "size": 16,
                    "align": "center",
                    "vadjust": -60
                }
            )

        # Ajouter le titre pour chaque couche
        if layer_idx == 0:
            title_text = "Input Layer"
        elif layer_idx == num_layers - 1:
            title_text = "Output Layer"
        else:
            title_text = f"Hidden Layer {layer_idx}"

        titles_y = -1010
        net.add_node(
            f"Title_{layer_idx}",
            label=title_text,
            x=layer_idx * x_spacing,
            y=titles_y,
            shape="text",
            font={"size": 20, "color": "black"}
        )

        # Ajouter le nombre de neurones pour chaque couche
        if n_neurons > 0:
            bottom_y = (n_neurons - 1) * y_spacing - y_offset + bottom_margin
            net.add_node(
                f"Bottom_{layer_idx}",
                label=str(n_neurons),
                x=layer_idx * x_spacing,
                y=bottom_y,
                shape="text",
                font={"size": 16, "color": "black"}
            )

    # 6) Créer les liens (edges) entre les neurones des différentes couches
    for layer_idx in range(num_layers - 1):
        from_neurons = sampled_indices[layer_idx]
        to_neurons = sampled_indices[layer_idx + 1]
        for i in from_neurons:
            for j in to_neurons:
                from_node = f"node_{layer_idx}_{i}"
                to_node = f"node_{layer_idx+1}_{j}"

                # Récupérer le poids pour chaque lien (edge)
                weight_label = ""
                if hasattr(mlp, "weights"):
                    try:
                        w_val = mlp.weights[layer_idx][i, j]
                        weight_label = f"w={w_val:.2f}"
                    except Exception:
                        pass

                # Ajouter l'edge au réseau
                edge_id = f"edge_{layer_idx}_{i}_{layer_idx+1}_{j}"
                net.add_edge(
                    from_node,
                    to_node,
                    id=edge_id,
                    title=weight_label,  # Poids affiché lors du clic
                    label="",            # Pas de label visible par défaut
                    hidden=False         # Liens visibles dès le départ
                )

    # 7) Désactiver la physique (pour un placement manuel)
    net.toggle_physics(False)

    # 8) Génération du fichier HTML
    html_content = net.generate_html(notebook=False)

    # 9) Ajouter le JavaScript pour afficher le poids du lien au clic
    custom_js = """
<script>
function clearAllEdgeLabels() {
    var edges = network.body.data.edges.get();
    edges.forEach(function(edge) {
        // Nous ne cachons pas les arêtes, nous effaçons seulement les labels
        edge.label = "";
        network.body.data.edges.update(edge);
    });
}

network.on("selectEdge", function(params) {
    clearAllEdgeLabels();
    if (params.edges.length === 1) {
        var selectedId = params.edges[0];
        var allEdges = network.body.data.edges.get();
        allEdges.forEach(function(edge) {
            if (edge.id === selectedId) {
                // Afficher le poids stocké dans edge.title
                edge.label = edge.title;
                network.body.data.edges.update(edge);
            }
        });
    }
});

network.on("deselectEdge", function(params) {
    clearAllEdgeLabels();
});
</script>
"""
    html_content = html_content.replace("</body>", custom_js + "\n</body>")

    # 10) Sauvegarde du fichier HTML final
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Interactive MLP visualization saved to", output_file)
    webbrowser.open(output_file)
