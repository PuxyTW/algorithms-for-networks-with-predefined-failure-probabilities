import networkx as nx
import os
import glob
import GML2GraphMLPlugin 

gml_files = list(glob.glob("./benchmark_graphs/*.gml"))

for gml_file in gml_files:
    try:
        graphml_file = os.path.splitext(gml_file)[0] + ".graphml"

        converter = GML2GraphMLPlugin()
        converter.input(gml_file)
        converter.run()
        converter.output(graphml_file)

        print(f"✅ Converted {gml_file} → {graphml_file}")

    except Exception as e:
        print(f"❌ Error converting {gml_file}: {e}")


"""
# Get all GML files in the directory
zoo_list = list(glob.glob("./benchmark_graphs/*.gml"))
print(f"Found {len(zoo_list)} GML files.")

for gml_file in zoo_list:
    try:
        # Define the output filename
        graphml_file = os.path.splitext(gml_file)[0] + ".graphml"

        # Load the GML file
        G = nx.read_gml(gml_file)

        # Ensure all node attributes are valid (convert dicts/lists/sets to strings)
        for node, data in G.nodes(data=True):
            for key in list(data.keys()):
                if isinstance(data[key], (dict, list, set, tuple)):
                    data[key] = str(data[key])  # Convert to string

        # Ensure all edge attributes are valid (convert dicts/lists/sets to strings)
        for u, v, data in G.edges(data=True):
            for key in list(data.keys()):
                if isinstance(data[key], (dict, list, set, tuple)):
                    data[key] = str(data[key])  # Convert to string

        # Handle duplicate node labels by renaming them
        seen_labels = set()
        for node in list(G.nodes()):
            if "label" in G.nodes[node]:  # Some GML files store labels in "label"
                label = G.nodes[node]["label"]
                if label in seen_labels:
                    new_label = f"{label}_{node}"  # Make the label unique
                    G.nodes[node]["label"] = new_label
                seen_labels.add(G.nodes[node]["label"])

        # Save as GraphML
        nx.write_graphml(G, graphml_file)

        print(f"Converted: {gml_file} -> {graphml_file}")

    except Exception as e:
        print(f"Error converting {gml_file}: {e}")

print("Batch conversion complete! 🎉")
"""