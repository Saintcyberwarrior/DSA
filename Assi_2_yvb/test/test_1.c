#include <stdio.h>
#include <stdlib.h>
struct Node {
	int data;
	struct Node* next;
};

void push(struct Node** a, int new_data){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = (*a);
        (*a) = new_node;
}

//void push(struct Node** head_ref, int new_data)
//{
    /* allocate node */
//    struct Node* new_node
//        = (struct Node*)malloc(sizeof(struct Node));

    /* put in the data */
//    new_node->data = new_data;

    /* link the old list off the new node */
//    new_node->next = (*head_ref);

    /* move the head to point to the new node */
//    (*head_ref) = new_node;
//}


int getCount(struct Node* head)
{
    int count = 1;  // Initialize count
    struct Node* current = head;  // Initialize current
    while (current != NULL)
    {
        if(current->next == NULL)
		return count;
	current = current->next;
        count++;
    }
    return count;
}
/*
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

       if we get to this line,
       the caller was asking
       for a non-existent element
       so we assert fail
   // assert(0);
   exit(1);
}

*/

int GetNth(struct Node* a, int i){
         struct Node* now;
         int count = 0;
         now = a;
         for(;now != NULL; count++){
                 if(count == i){
                 return now->data;
                 }
                 now = now->next;
	 }
         fprintf(stderr,"Over the limit\n");
         return 1;
 }
int max(int x, int y){
	if (x >=y)
		return x;
	else
		return y;
}

int min(int x, int y){
	if (x<=y)
		return x;
	else
		return y;
}

void frre(struct Node* begin){
         struct Node* now= begin ;
         while(now != NULL && begin->next !=NULL){
                 now = begin->next;
                 free(begin);
                 begin=now;
         }
 }

struct Node* add(struct Node* a, struct Node* b){
	int len_a = getCount(a);
	int len_b = getCount(b);
	struct Node* c = NULL;

	int i = 0;
	int car=0;
	push(&c, (GetNth(a, len_a-1)+ GetNth(b, getCount(b)-1))% 1000);
	car = (GetNth(a, len_a-1)+GetNth(b, getCount(b)-1))/1000;

	for (; i < max(len_a, getCount(b)); i++){
		if(i < min(len_a, getCount(b))-1){ //this might be wrong
			push(&c, ((GetNth(a, len_a-2-i)+ GetNth(b,getCount(b)-2-i)) + car)%1000);
			car = ((GetNth(a, len_a-2-i)+GetNth(b, getCount(b)-2-i))+car)/1000;
		}else {
			if(len_a<= i && i < getCount(b)){
				push(&c, ((GetNth(b, getCount(b)-1-i)+car)%1000));
				car = ((GetNth(b, getCount(b)-1-i)+car)/1000);
			}else{
				push(&c, ((GetNth(a, len_a-1-i)+car)%1000));
				car = ((GetNth(a, len_a-1-i)+car)/1000);
			}
		}
	}
	push(&c, car);
	return c;
}
int main(void){
	printf("1\n");
	struct Node* p = NULL;
	push(&p, 555);
	push(&p, 465);
	push(&p, 564);
	push(&p, 597);
	struct Node* q = NULL;
	push(&q, 295);
	push(&q, 565);
	push(&q, 581);
	struct Node *r = NULL;
	r = add(p, q);

	printf("%d\n", getCount(r));

	printf("%d\n", GetNth(r, 0));
	printf("%d\n", GetNth(r, 1));
	printf("%d\n", GetNth(r, 2));
	printf("%d\n", GetNth(r, 3));
	printf("%d\n", GetNth(r, 4));
	printf("%d\n", GetNth(r, 5));


	frre(p);
	frre(q);
	printf("\nSuccess!!\n");
	return 0;
}

