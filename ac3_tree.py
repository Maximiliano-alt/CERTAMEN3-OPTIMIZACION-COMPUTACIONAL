from graphviz import Digraph

g = Digraph(format='pdf')
g.attr(rankdir='LR', fontname='Helvetica', fontsize='10')

g.node('x4', 'x4\n{0‒4}')
g.node('x3', 'x3\n{0‒25}')
g.node('x2', 'x2\n{0‒10}')
g.node('x1', 'x1\n{0‒15}')
g.node('x5', 'x5\n{0‒30}')

g.edge('x4', 'x3'); g.edge('x3', 'x2'); g.edge('x2', 'x1'); g.edge('x1', 'x5')
g.render('ac3_tree')      # crea ac3_tree.pdf y ac3_tree.png
