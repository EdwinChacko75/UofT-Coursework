#include <stdlib.h>
#include <stdio.h>

typedef struct Con{
    double weight;
    struct Node *node;
} Con;

typedef struct Node {
    int data;
    int num_cons;
    struct Con **connections;
    double dist_from_start;
} Node;


void create_node(int data, Node **p_node) {
    *p_node = malloc(sizeof(Node));
    (*p_node)->data = data;
    (*p_node)->connections = (void *)0;
    (*p_node)->num_cons = 0;
    (*p_node)->dist_from_start = 10000000.0;
}

void add_neighbour(Node *node, Node *neighbour, double weight) {
    node->num_cons++;
    node->connections = realloc(node->connections, node->num_cons * sizeof(Con *));
    node->connections[node->num_cons - 1] = malloc(sizeof(Con));
    node->connections[node->num_cons - 1]->weight = weight;
    node->connections[node->num_cons - 1]->node = neighbour;
}

void add_to_visited(Node ***p_visited, int *p_n_visited, Node *node) {
    Node **visited = *p_visited;

    for(int i = 0; i < *p_n_visited; i++) {
        if (visited[i] == node) {
            return;
        }
    }
    (*p_n_visited)++;
    visited = realloc(visited, *p_n_visited * sizeof(Node *));
    visited[*p_n_visited - 1] = node;

    *p_visited = visited;
}

int was_visited(Node **visited, int n_visited, Node *node) {
    for(int i = 0; i < n_visited; i++) {
        if (visited[i] == node) {
            return 1;
        }
    }
    return 0;
}

void dijkstra(Node *start, Node *end) {
    Node **visited = (void *)0;
    int num_visited = 0;
    Node *current = start;
    add_to_visited(&visited, &num_visited, current);

    while (current != end) {
        double min_dist = 1000000000.0;
        Node *min_node = (void *)0;
        for(int i = 0; i < num_visited; i++){
            Node *s = visited[i];
            for(int j = 0; j < s->num_cons; j++) {
                Node *neighbour = s->connections[j]->node;
                if (was_visited(visited, num_visited, neighbour)) {
                    continue;
                }
                double dist = s->dist_from_start + s->connections[j]->weight;
                if (dist < min_dist) {
                    min_dist = dist;
                    min_node = neighbour;
                }
            }
        }
        current = min_node;
        current->dist_from_start = min_dist;
        add_to_visited(&visited, &num_visited, current);
    }
}

int main() {
    Node *node1, *node2, *node3, *node4, *node5;
    create_node(1, &node1);
    create_node(2, &node2);
    create_node(3, &node3);
    create_node(4, &node4);
    create_node(5, &node5);

    // Add connections
    add_neighbour(node1, node2, 5.0);
    add_neighbour(node1, node3, 2.0);
    add_neighbour(node2, node3, 1.0);
    add_neighbour(node2, node4, 6.0);
    add_neighbour(node3, node4, 1.0);
    add_neighbour(node3, node5, 4.0);
    add_neighbour(node4, node5, 2.0);

    // Find shortest path
    dijkstra(node1, node5);

    // Print shortest distance
    printf("Shortest distance from node %d to node %d: %f\n", node1->data, node5->data, node5->dist_from_start);

    return 0;
}