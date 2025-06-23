
import flowkit as fk
import pandas as pd
import os
from lxml import etree as ET
import re

def parse_fcs_add_gate_label(wsp_path, wsp_fcs_dir, csv_dir):
    """ Uses the converted and processed .csv files and the .fcs files that they origin from. 
    Takes the Gate_Label from the .fcs file and applies it to the .csv """

    workspace = fk.Workspace(wsp_path, fcs_samples = wsp_fcs_dir)
    workspace.analyze_samples()
    sample_ids = list(workspace.get_sample_ids())
    print(f"Found {len(sample_ids)} samples to process")

    for sample_id in sample_ids:
        print(f"Processing {sample_id}")

        try:
            sample = workspace.get_sample(sample_id)
            gate_ids = workspace.get_gate_ids(sample_id)
            terminal_gates = []
            for gate_id in gate_ids:
                child_gates = workspace.get_child_gate_ids(sample_id, gate_id[0], gate_path= gate_id[1])
                if not child_gates:  # No children = terminal gate
                    terminal_gates.append(gate_id)
            
            # Get the number of events in the sample - each event needs a gate label
            num_events = len(sample.get_events(source= 'raw'))
            
            # Initialize gate labels with "Ungated"
            gate_labels = ["Ungated"] * num_events

            for gate_id in terminal_gates:
                try:
                    membership = workspace.get_gate_membership(sample_id, gate_id[0], gate_path= gate_id[1])
                    # Update labels for cells in this terminal gate
                    for i, is_member in enumerate(membership):
                        if is_member:
                            gate_labels[i] = gate_id
                except Exception as e:
                    print(f"Error getting membership for terminal gate {gate_id}: {e}")

            gate_df = pd.DataFrame({'Gate_Label': gate_labels})

            sample_id_nofcs = sample_id.replace('.fcs','')
            csv_path = os.path.join(csv_dir, f"{sample_id_nofcs}.csv")
            if not os.path.exists(csv_path):
                print(f"CSV for {sample_id} not found, skipping.")
                continue

            csv_df = pd.read_csv(csv_path)

            if len(csv_df) != len(gate_df):
                print(f"Mismatch in cell count for {sample_id}: CSV has {len(csv_df)}, FCS has {len(gate_df)}")
                continue

            # Add the Gate_Label to your CSV
            csv_df["Gate_Label"] = gate_df["Gate_Label"].values

            # Save output

            output_path = os.path.join(csv_dir, f"{sample_id_nofcs}_with_gate_label.csv")
            csv_df.to_csv(output_path, index=False)
            print(f"Saved labeled CSV for {sample_id_nofcs}")

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
    print('.csv processing complete')
    return None

def extract_gating_strategy(wsp_path, wsp_fcs_dir, output_path="./gating_structure.csv"):
    """Extract gating strategy from FlowJo workspace and save as a CSV for UNITO to use
    Here use a LCRS approach to reduce the N - Tree of a gating strategy to a binary tree. """

    class GateNode:
        def __init__(self, name, x_axis = None, y_axis = None, parent = None):
            self.name = name
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.parent = parent
            self.children = []
            self.left_child = None
            self.right_sibling = None
    
    def _parse_flowjo_workspace_flowkit(wsp_path, wsp_fcs_dir):
        # load and analyze workspace
        ws = fk.Workspace(wsp_path, fcs_samples=wsp_fcs_dir)
        ws.analyze_samples()

        sample_ids = ws.get_sample_ids()
        if not sample_ids:
            print("No samples found in workspace")
            return []
        sample_id = sample_ids[0]

        # get all gate IDs (tuple of (gate_id, gate_path))
        gate_ids = ws.get_gate_ids(sample_id)
        print(f"Found {len(gate_ids)} gates via FlowKit")

        gate_nodes = {}
        root_gates = []

        # create GateNode for each gate
        for gate_id, gate_path in gate_ids:
            g = ws.get_gate(sample_id, gate_id, gate_path=gate_path)
            dims = getattr(g, "dimensions", [])
            x = dims[0] if len(dims) >= 1 else None
            y = dims[1] if len(dims) >= 2 else ("Histogram" if len(dims) == 1 else None)
            node = GateNode(gate_id, x_axis=x, y_axis=y)
            gate_nodes[(gate_id, gate_path)] = node

        # Link parent/child using FlowKit
        for gate_id, gate_path in gate_ids:
            parent_node = gate_nodes[(gate_id, gate_path)]
            child_ids = ws.get_child_gate_ids(sample_id, gate_id, gate_path=gate_path)
            for child_id, child_path in child_ids:
                child_node = gate_nodes[(child_id, child_path)]
                parent_node.children.append(child_node)
                child_node.parent = parent_node

        root_gates = [n for n in gate_nodes.values() if n.parent is None]

        print(f"Built hierarchy with {len(root_gates)} root gates")
        return root_gates

    def _convert_to_lcrs(root_gates):
        """Convert N-ary tree to Left Child Right Sibling representation"""
        def _convert_node_(node):
            if node.children:
                node.left_child = node.children[0]
                for i in range(len(node.children) - 1):
                    node.children[i].right_sibling = node.children[i+1]
                for child in node.children:
                    _convert_node_(child)
            return node

        return[_convert_node_(r) for r in root_gates] 

    def _get_all_paths_lcrs(lcrs_roots):
        """ Get all possible root to leaf paths from the LCRS tree """
        paths = []
        def _path_extract_(node, path):
            path.append(node)
            if node.left_child is None:
                paths.append(path.copy())
            else:
                _path_extract_(node.left_child, path)
            path.pop()
            if node.right_sibling:
                _path_extract_(node.right_sibling, path)
        for r in lcrs_roots:
            _path_extract_(r,[])
        return paths

    def _generate_strategy_dataframe(all_paths):
        data = []
        for pid, path in enumerate(all_paths):
            pname = " -> ".join(n.name for n in path)
            for sid, node in enumerate(path):
                data.append({
                    'Gate': node.name,
                    'Parent_Gate': path[sid-1].name if sid>0 else 'None',
                    'X_axis': node.x_axis,
                    'Y_axis': node.y_axis,
                    'Is_Terminal': sid==len(path)-1,
                    'Path_ID': pid,
                    'Path_Name': pname,
                    'Step': sid
                })
        return pd.DataFrame(data)

# Code to run functions
    print(f"Parsing FlowJo workspace: {wsp_path}")
    roots = _parse_flowjo_workspace_flowkit(wsp_path, wsp_fcs_dir)
    if not roots:
        print("No gates found")
        return None
    lcrs = _convert_to_lcrs(roots)
    print("Converted to LCRS")
    paths = _get_all_paths_lcrs(lcrs)
    print(f"Extracted {len(paths)} paths")
    df = _generate_strategy_dataframe(paths)

    # Dataframe cleanup
    df = df.drop_duplicates(subset=['Gate','Parent_Gate'], keep='first') #UNITO needs just one entry of a gate - this removes duplicates of earlier nodes created by the LCRS approach.

    return df

def clean_gating_strategy(panel_metadata_path, gating_strat_df = None):
    """ Required in the absence of proper cytometry labels - uses batch correction panel file to correct X and Y axis labels from flurophore -> marker """
    panel_metadata = pd.read_csv(panel_metadata_path)
    gating_strat_df = gating_strat_df.copy()

    for col in ['X_axis', 'Y_axis']:
        gating_strat_df[col] = gating_strat_df[col].astype(str).str.extract(r'id:\s*([^)]*)', expand=False).str.strip()
                                
    channel_to_antigen = panel_metadata.set_index('Channel')['Antigen'].to_dict()
    print(gating_strat_df)

    gating_strat_df['X_axis'] = gating_strat_df['X_axis'].map(channel_to_antigen).fillna(gating_strat_df['X_axis'])
    gating_strat_df['Y_axis'] = gating_strat_df['Y_axis'].map(channel_to_antigen).fillna(gating_strat_df['Y_axis'])

    return gating_strat_df
    