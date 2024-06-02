import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

# Nodes
G.add_node("Define Sphere\n(R=64)")
G.add_node("Define Window\n(Size=15)")
G.add_node("Create Sphere Wireframe\n(using phi, theta)")
G.add_node("Define Window Coordinates\n(Corners)")
G.add_node("Person 1\n(0,0,0)")
G.add_node("Person 2\n(dx,0,0)")
G.add_node("Calculate Intersection Points\n(Equation and Discriminant)")
G.add_node("Visualize Final Result\n(Sphere, Window, Positions, Lines of Sight)")

# Edges
G.add_edges_from([
    ("Define Sphere\n(R=64)", "Create Sphere Wireframe\n(using phi, theta)"),
    ("Define Window\n(Size=15)", "Define Window Coordinates\n(Corners)"),
    ("Define Window Coordinates\n(Corners)", "Person 1\n(0,0,0)"),
    ("Define Window Coordinates\n(Corners)", "Person 2\n(dx,0,0)"),
    ("Person 1\n(0,0,0)", "Calculate Intersection Points\n(Equation and Discriminant)"),
    ("Person 2\n(dx,0,0)", "Calculate Intersection Points\n(Equation and Discriminant)"),
    ("Calculate Intersection Points\n(Equation and Discriminant)", "Visualize Final Result\n(Sphere, Window, Positions, Lines of Sight)")
])

# Plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, font_weight="bold", edge_color="gray", linewidths=2, width=2, arrows=True)
plt.title("Code Flow for Sphere and Window Intersection")
plt.show()
