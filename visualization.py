# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_network(layer_sizes, X, biases, filename="network_vis.png", folder="pictures"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if X.ndim > 1 and X.shape[1] != 1:
        X = X.flatten()
    positions = {}
    neuron_counter = 0
    x_spacing = 2   # horizontal spacing
    y_spacing = 1   # vertical spacing
    for layer_idx, num_neurons in enumerate(layer_sizes):
        total_height = (num_neurons - 1) * y_spacing
        for neuron_idx in range(num_neurons):
            x = layer_idx * x_spacing
            y = neuron_idx * y_spacing - total_height/2
            positions[neuron_counter] = (x, y)
            neuron_counter += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    neuron_counter = 0
    for layer in range(len(layer_sizes) - 1):
        current_layer_neurons = layer_sizes[layer]
        next_layer_neurons = layer_sizes[layer+1]
        for i in range(current_layer_neurons):
            start_pos = positions[neuron_counter + i]
            for j in range(next_layer_neurons):
                end_pos = positions[neuron_counter + current_layer_neurons + j]
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', lw=1)
        neuron_counter += current_layer_neurons
    for pos in positions.values():
        circle = plt.Circle(pos, 0.2, color='lightblue', ec='black', zorder=3)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Neural Network Visualization")
    out_path = os.path.join(folder, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Network visualization image saved to: {os.path.abspath(out_path)}")

def generate_interactive_html(layer_sizes, X, weights, biases, filename="network.html", folder="html"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    html_path = os.path.join(folder, filename)
    max_neurons = max(layer_sizes)
    vertical_spacing = 100
    horizontal_spacing = 200
    positions = {}
    neuron_details = {}
    node_id = 0
    colors = []
    for idx in range(len(layer_sizes)):
        if idx == 0:
            colors.append("#4CAF50")
        elif idx == len(layer_sizes)-1:
            colors.append("#FF5722")
        else:
            colors.append("#2196F3")
    if X.ndim > 1 and X.shape[1] != 1:
        X = X.flatten()
    for layer_idx, size in enumerate(layer_sizes):
        x = layer_idx * horizontal_spacing + 50
        y_start = (max_neurons - size) * vertical_spacing / 2
        for neuron_idx in range(size):
            y = y_start + neuron_idx * vertical_spacing
            positions[node_id] = (x, y)
            if layer_idx == 0:
                try:
                    val = float(X[neuron_idx])
                    detail = f"Input: {val:.4f}"
                except:
                    detail = "Input: N/A"
            else:
                try:
                    bias_val = float(biases[layer_idx-1][neuron_idx])
                    detail = f"Bias: {bias_val:.4f}"
                except:
                    detail = "Bias: N/A"
            neuron_details[node_id] = {"layer": layer_idx, "index": neuron_idx, "detail": detail, "color": colors[layer_idx]}
            node_id += 1
    connections = []
    for l in range(len(weights)):
        cur_size = layer_sizes[l]
        nxt_size = layer_sizes[l+1]
        for i in range(cur_size):
            for j in range(nxt_size):
                try:
                    w_val = float(weights[l][j, i])
                except:
                    w_val = 0.0
                start_id = sum(layer_sizes[:l]) + i
                end_id = sum(layer_sizes[:l+1]) + j
                x1, y1 = positions[start_id]
                x2, y2 = positions[end_id]
                line_color = "#006400" if w_val >= 0 else "#8B0000"
                line = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="{line_color}" stroke-width="2" '
                        f'onclick="showWeight({l},{i},{l+1},{j},{w_val})" />')
                connections.append(line)
    neurons = []
    for node, (x, y) in positions.items():
        detail = neuron_details[node]["detail"]
        color = neuron_details[node]["color"]
        layer = neuron_details[node]["layer"]
        idx = neuron_details[node]["index"]
        circle = (f'<circle cx="{x}" cy="{y}" r="20" fill="{color}" stroke="black" stroke-width="1" '
                  f'onclick="showDetail({layer},{idx},\'{detail}\')"/>'
                  f'<text x="{x}" y="{y+5}" font-size="10" text-anchor="middle" fill="black">'
                  f'{layer+1}-{idx+1}</text>')
        neurons.append(circle)
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Neural Network Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #f0f0f0; margin: 20px; }}
        #info {{ margin-bottom: 20px; padding:10px; background-color:#fff; border:1px solid #ccc; }}
        .legend div {{ margin:5px 0; }}
        .legend span {{ display:inline-block; width:15px; height:15px; margin-right:5px; vertical-align:middle; }}
        svg {{ margin-top: 50px; margin-bottom: 50px; }}
    </style>
    <script>
        function showDetail(layer, neuron, detail) {{
            document.getElementById('info').innerHTML =
                'Layer ' + (layer+1) + ', Neuron ' + (neuron+1) + ': ' + detail;
        }}
        function showWeight(fromLayer, fromNeuron, toLayer, toNeuron, weight) {{
            document.getElementById('info').innerHTML =
                'Weight: from Layer ' + (fromLayer+1) + '-' + (fromNeuron+1)
                + ' to Layer ' + (toLayer+1) + '-' + (toNeuron+1)
                + ' = ' + weight.toFixed(4);
        }}
        function clearInfo() {{
            document.getElementById('info').innerHTML = '';
        }}
    </script>
</head>
<body>
    <h1>Interactive Neural Network Visualization</h1>
    <div class="legend">
        <strong>Legend:</strong>
        <div><span style="background-color:#4CAF50;"></span> Input Layer</div>
        <div><span style="background-color:#2196F3;"></span> Hidden Layer(s)</div>
        <div><span style="background-color:#FF5722;"></span> Output Layer</div>
        <div><span style="stroke:#006400; stroke-width:2px;"></span> Positive Weight</div>
        <div><span style="stroke:#8B0000; stroke-width:2px;"></span> Negative Weight</div>
    </div>
    <div id="info" style="margin-bottom:20px; padding:10px; background-color:#fff; border:1px solid #ccc;" onclick="clearInfo()">
        Click on a neuron or connection for details. Click here to clear.
    </div>
    <svg width="{len(layer_sizes)*horizontal_spacing + 200}" height="{max_neurons*vertical_spacing + 200}">
        {"".join(connections)}
        {"".join(neurons)}
    </svg>
</body>
</html>'''
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Interactive HTML saved to: {os.path.abspath(html_path)}")
