#include <stdio.h>
#include <stdlib.h>
struct Node {
	int data;
	struct Node* next;
};

void push(struct Node* a, int new_data){
	//allocation of memory
	struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
	//substitute new data in the memory created
	new_node->data = new_data;
	//link this node in the front of already existing linked list
	new_node->next = a;
	//now, new linked list is the one with this node as well
	a = new_node;
}

int main(void){
	struct Node* p;
	p->next = NULL;
	push(p, 23);
	push(p, 32);
	push(p, 45);
	push(p, 89);
	push(p, 26);
	printf("Done");
	return 0;
}
