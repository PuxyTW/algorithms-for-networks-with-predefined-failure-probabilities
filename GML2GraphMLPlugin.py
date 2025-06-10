import glob
import os

class GML2GraphMLPlugin:
    def __init__(self):
        self.in_gml = None

    def input(self, infile):
        self.in_gml = infile

    def run(self):
        pass  # No processing needed here

    def output(self, outfile):
        out_graphml = outfile

        graph_id = "GraphML_Output"

        graphml_header = '''<?xml version="1.0" encoding="UTF-8"?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
        http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd">
        <key id="key_weight" for="edge" attr.name="weight" attr.type="double"/>
        <graph id="{}" edgedefault="directed">
        '''.format(graph_id)

        # Read GML and process
        graph_dict = {}
        with open(self.in_gml, "r", encoding="utf-8") as f_gml:
            node = False
            edge = False
            source, target, weight = None, None, None  # Ensure variables are defined

            for line in f_gml:
                line = line.strip()
                if line == "node [":
                    node = True
                elif line == "edge [":
                    edge = True
                elif line == "]":
                    node = False
                    edge = False

                if node:
                    if line.startswith("id "):
                        id = line.split(" ", 1)[1].strip('"')
                        graph_dict[id] = {"edges": []}
                    elif line.startswith("label "):
                        label = line.split(" ", 1)[1].strip('"')
                        graph_dict[id]["label"] = label
                elif edge:
                    if line.startswith("source "):
                        source = line.split(" ", 1)[1].strip('"')
                    elif line.startswith("target "):
                        target = line.split(" ", 1)[1].strip('"')
                    elif line.startswith("dist "):
                        weight = line.split(" ", 1)[1].strip('"')
                        if source in graph_dict and target in graph_dict:
                            graph_dict[source]["edges"].append((graph_dict[target]["label"], weight))
        print(f"Checking edges in {self.in_gml}...")
        for id in graph_dict.keys():
            if "edges" in graph_dict[id] and graph_dict[id]["edges"]:
                print(f"  Node {graph_dict[id]['label']} has edges: {graph_dict[id]['edges']}")

        # Write GraphML file
        with open(out_graphml, "w", encoding="utf-8") as f_graphml:
            f_graphml.write(graphml_header)

            # Write nodes
            for id, data in graph_dict.items():
                if "label" in data:
                    f_graphml.write(f'<node id="{data["label"]}"/>\n')

            # Write edges
            i = 1
            for id, data in graph_dict.items():
                for e, edge in enumerate(data["edges"]):
                    target_label, weight = edge
                    f_graphml.write(f'<edge id="e{i}" source="{data["label"]}" target="{target_label}">\n')
                    f_graphml.write(f'<data key="key_weight">{weight}</data>\n')
                    f_graphml.write('</edge>\n')
                    i += 1

            f_graphml.write("</graph></graphml>")  # Close tags

        print(f"✅ Successfully converted {self.in_gml} → {out_graphml}")


# Convert all .gml files
gml_files = glob.glob("./benchmark_graphs/*.gml")

for gml_file in gml_files:
    try:
        graphml_file = os.path.splitext(gml_file)[0] + ".graphml"

        converter = GML2GraphMLPlugin()  # Ensure correct initialization
        converter.input(gml_file)
        converter.run()
        converter.output(graphml_file)

    except Exception as e:
        print(f"❌ Error converting {gml_file}: {e}")
