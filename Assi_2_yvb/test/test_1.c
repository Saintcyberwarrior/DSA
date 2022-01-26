#include <stdio.h>
struct Node {
	int data;
	struct Node* next;
};

void push(struct Node** head_ref, int new_data)
{
    /* allocate node */
    struct Node* new_node
        = (struct Node*)malloc(sizeof(struct Node));

    /* put in the data */
    new_node->data = new_data;

    /* link the old list off the new node */
    new_node->next = (*head_ref);

    /* move the head to point to the new node */
    (*head_ref) = new_node;
}

int getCount(struct Node* head)
{
    int count = 0;  // Initialize count
    struct Node* current = head;  // Initialize current
    while (current != NULL)
    {
        count++;
        current = current->next;
    }
    return count;
}

int GetNth(struct Node* head, int index)
{

    struct Node* current = head;

    // the index of the
    // node we're currently
    // looking at
    int count = 0;
    while (current != NULL) {
        if (count == index)
            return (current->data);
        count++;
        current = current->next;
    }

    /* if we get to this line,
       the caller was asking
       for a non-existent element
       so we assert fail */
    assert(0);
}


int add(struct Node a, struct Node b){
	int len_a = getCount(a);
	int len_b = getCount(b);
	struct Node* sum = NULL;

	int i = 0;
	push(&c, (GetNth(a, getCount(a)--)+ GetNth(b, getCount(b)--))% 1000);

	if(i<=min(getCount(a), getCount(b))){
		int car = (GetNth(a, getCount(a)-1-i)+GetNth(b, getCount(b)-1-i))/1000;
		push(&c, (GetNth(a, getCount(a)-2-i)+ getCount(b)-2-i)%1000 + car);
		i += 1;
	}elseif(){
		push(&c, )

	}
	
}
int main(void){
	
	return 0;
}


