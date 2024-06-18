from matplotlib import pyplot as plt
import numpy as np


def compute_eco_return(id_to_parent_id, discount_factor=0.99):
    # Step 1: Identify leaf agents
    root_ids = {id for id in id_to_parent_id.values() if id != -1}
    all_ids = set(id_to_parent_id.keys()).union(root_ids)
    leaf_ids = all_ids - root_ids

    # Initialize id_to_eco_return with 1 for leaf agents
    id_to_eco_return = {agent_id: 0 for agent_id in leaf_ids}

    def compute_score(agent_id):
        # If the score is already computed, return it
        if agent_id in id_to_eco_return:
            return id_to_eco_return[agent_id]

        # Compute score for non-leaf agent
        children_scores = []
        children_count = 0
        for child_id, parent_id in id_to_parent_id.items():
            if parent_id == agent_id:
                children_scores.append(compute_score(child_id))
                children_count += 1

        score = children_count + discount_factor * sum(children_scores)

        id_to_eco_return[agent_id] = score
        return score

    # Step 3: Compute the score for each agent including the root nodes
    for agent_id in all_ids:
        if agent_id not in id_to_eco_return:
            compute_score(agent_id)

    return id_to_eco_return


def get_phylogenetic_tree(id_to_parent_id, id_to_timestep_born):

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Dictionary to store positions of nodes
    positions = {}
    
    # Function to recursively plot tree
    def plot_tree(id_parent, x_offset):
        list_id_childs = [id_child for id_child, p in id_to_parent_id.items() if p == id_parent]
        num_children = len(list_id_childs)
        
        # Calculate y-position based on birth timestep
        if id_parent in id_to_timestep_born:
            timestep_parent_born = id_to_timestep_born[id_parent]
        else:
            timestep_parent_born = 0
        
        y_position = timestep_parent_born  # Higher y for earlier birth
        
        # Plot each child node and recursively plot its subtree
        for i, id_child in enumerate(list_id_childs):
            # Calculate x-position based on number of children and order, with added randomness
            child_x_offset = x_offset + (i - (num_children - 1) / 2) * 2.0 + np.random.uniform(-0.5, 0.5)
        
            # Store position of child node
            positions[id_child] = (child_x_offset, y_position)
            
            if id_parent != -1:
                # Plot edge from parent to child
                ax.plot([positions[id_parent][0], child_x_offset], [positions[id_parent][1], y_position], 'k-')
                
                # Plot child node
                ax.plot(child_x_offset, y_position, 'ko', markersize=12)
            
            # Recursively plot subtree
            plot_tree(id_child, child_x_offset)
    
    # Find root node(s)
    roots = [node_id for node_id, parent_id in id_to_parent_id.items() if parent_id == -1]
    
    # If there are multiple roots, create a virtual root
    if len(roots) > 1:
        virtual_root = -1
        positions[virtual_root] = (0, len(id_to_timestep_born) + 1)
        for root in roots:
            id_to_parent_id[root] = virtual_root
        plot_tree(virtual_root, 0)

    else:
        # Start plotting from the actual root node
        root = roots[0]
        positions[root] = (0, len(id_to_timestep_born) + 1)
        plot_tree(root, 0)

    
    # Annotate nodes with IDs
    for node_id, (x, y) in positions.items():
        ax.text(x, y, str(node_id), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    
    # Set plot title and axis labels
    ax.set_title('Phylogenetic Tree', fontsize=15)
    ax.set_ylabel('timestep of birth')
    
    # Adjust margins and show plot
    ax.margins(0.1)
    plt.tight_layout()
    # plt.show()
    return fig




if __name__ == "__main__":
    # Example usage

    id_to_parent_id = {0: -1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4}
    id_to_timestep_born = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5}
    eco_return = compute_eco_return(id_to_parent_id)
    print(eco_return)
    
    get_phylogenetic_tree(id_to_parent_id, id_to_timestep_born)
