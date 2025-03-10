import os
import numpy as np

def generate_network_visualization(weights, layer_sizes, output_filename="MLP_Visualization.html"):
    """
    Generates an HTML file visualizing the network architecture and weight matrices.
    Each weight matrix is shown as a color-coded table (blue for positive, red for negative).
    """
    html_lines = []
    html_lines.append("<html><head><title>MLP Network Visualization</title>")
    html_lines.append("<style>")
    html_lines.append("table { border-collapse: collapse; }")
    html_lines.append("td { padding: 2px; font-size: 10px; text-align: center; }")
    html_lines.append("</style></head><body>")
    html_lines.append("<h2>MLP Network Visualization</h2>")
    arch_text = " → ".join(str(s) for s in layer_sizes)
    html_lines.append(f"<p><b>Architecture:</b> {arch_text}</p>")
    
    for i, W in enumerate(weights):
        # Determine layer names
        if i == 0:
            from_layer = "Input"
            to_layer = "Hidden"
        elif i == len(weights)-1:
            from_layer = "Hidden"
            to_layer = "Output"
        else:
            from_layer = "Hidden"
            to_layer = "Hidden"
        html_lines.append(f"<h3>Weights: {from_layer} → {to_layer}</h3>")
        html_lines.append("<table border='1'>")
        # Transpose for easier display: rows = neurons in from-layer, cols = neurons in to-layer
        W_t = W.T
        max_val = np.max(np.abs(W_t))
        if max_val == 0:
            max_val = 1e-6
        for row in range(W_t.shape[0]):
            html_lines.append("<tr>")
            for col in range(W_t.shape[1]):
                val = W_t[row, col]
                norm = val / max_val
                # Determine background color: positive -> blue, negative -> red.
                if norm >= 0:
                    intensity = int(255 * (1 - norm))
                    bg_color = f"rgb({intensity}, {intensity}, 255)"
                else:
                    intensity = int(255 * (1 - abs(norm)))
                    bg_color = f"rgb(255, {intensity}, {intensity})"
                html_lines.append(f"<td style='background-color: {bg_color};'>{val:.2f}</td>")
            html_lines.append("</tr>")
        html_lines.append("</table>")
    html_lines.append("</body></html>")
    with open(output_filename, "w") as f:
        f.write("\n".join(html_lines))
    print(f"Network visualization saved to: {os.path.abspath(output_filename)}")
