#include <stdio.h>
#include <stdlib.h>

struct Node {
	int data;
	struct Node* next;
};

void fpush(struct Node** a, int new_data);
void bpush(struct Node** a, int new_data);
int getCount(struct Node* head);
int GetNth(struct Node* a, int i);
int max(int x, int y);
int min(int x, int y);
void frre(struct Node* begin);
struct Node* add(struct Node* a, struct Node* b);
struct Node* mul(struct Node* a, struct Node* b);
void print(struct Node *temp);
struct Node* parse(char input[]);
int converter(char);
int calcnumber(int a[],int j);

int main(){
	printf("Enter an empty = to quit the program\n");
	char input[800];
	int i = 0;
	while((input[i] = getchar())){
		if(input[i] == '=')
			break;
		i++;
		if(i==799){
			printf("String too long\n");
			return -1;
		}
	}
	struct Node* res = NULL;
	res = parse(input);
	print(res);
	frre(res);
/*	struct Node* p = NULL;
	fpush(&p, 999);
	fpush(&p, 996);
	struct Node* q = NULL;
	fpush(&q, 111);
	fpush(&q, 999);
	struct Node *s = NULL;
	struct Node *r = NULL;
	s = add(q,p);
	r = mul(p,q);
	print(p);
	print(q);
	print(r);
	print(s);

	frre(p);
	frre(q);
	frre(r);
	frre(s);
*/
	
	return 0;
}

void print(struct Node *temp){
	while(temp){
		if(!(temp->next)){
			printf("%d$\n",temp->data);
			return;
		}
		printf("%d,",temp->data);
		temp = temp->next;
	}
}


void fpush(struct Node** a, int new_data){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = (*a);
        (*a) = new_node;
}

void bpush(struct Node** a, int new_data){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = NULL;
	if((*a)==NULL){
		(*a) = new_node;
		return;
	}
	struct Node* temp = (*a);
	while(temp->next){
		temp = temp->next;
	}
	temp->next = new_node;
}

int getCount(struct Node* head){
	int count = 1;  // Initialize count
	struct Node* current = head;  // Initialize current
	while (current != NULL){
		if(current->next == NULL)
			return count;
		current = current->next;
		count++;
	}
	return count;
}

int GetNth(struct Node* a, int i){
         struct Node* now;
         int count = 1;
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

	if(!a)
		return b;
	if(!b)
		return a;

	int car = 0;

	for(int i = 0; i < max(len_a, len_b); i++){
		if(i < min(len_a, len_b)){
			fpush(&c, (GetNth(a, len_a-i) + GetNth(b, len_b-i) + car)%1000);
			car = (GetNth(a, len_a-i) + GetNth(b, len_b-i) + car)/1000;
		}else{
			if(len_b>i){
				fpush(&c, (GetNth(b, len_b-i) + car)%1000);
				car = (GetNth(b, len_b-i) + car)/1000;
			}else{
				fpush(&c, (GetNth(a, len_a-i) + car)%1000);
				car = (GetNth(a, len_a-i) + car)/1000;
			}
		}
	}
	if(car)
		fpush(&c, car);
	return c;
}


struct Node* mul(struct Node* a, struct Node *b){
        int len_a = getCount(a);
        int len_b = getCount(b);

        struct Node* t_0 = NULL;
        struct Node* t_1 = NULL;

        if(len_a<len_b){
                int car = 0;

                for(int j=0; j<len_b;j++){
                        fpush(&t_0,((GetNth(a, len_a)*GetNth(b, len_b-j))+car)%1000);
                        car = ((GetNth(a, len_a) * GetNth(b, len_b-j)) + car)/1000;
                }
                if(car)
                        fpush(&t_0, car);

                for (int i =1; i < len_a; i++){
                        int car =0;

                        for(int j= 0; j<len_b; j++){
                                fpush(&t_1, ((GetNth(a, len_a-i)*GetNth(b, len_b-j))+car)%1000);
                                car = ((GetNth(a, len_a-i) * GetNth(b, len_b-j)) + car)/1000;
                        }
                        for (int count=1; count<i; count++){
                                bpush(&t_1, 000);
                        }


                        if(car)
                                fpush(&t_1, car);

                        t_0 = add(t_0, t_1);
                 }

                 frre(t_1);
		
                return t_0;
        }
        else{
                int car = 0;

                for(int j=0; j<len_a;j++){
                        fpush(&t_0,((GetNth(b, len_b)*GetNth(a, len_a-j))+car)%1000);
                        car = ((GetNth(b, len_b) * GetNth(a, len_a-j)) + car)/1000;
                }
                if(car)
                        fpush(&t_0, car);

                for (int i =1; i < len_b; i++){
                        int car =0;

                        for(int j= 0; j<len_a; j++){
                                fpush(&t_1, ((GetNth(b, len_b-i)*GetNth(a, len_a-j))+car)%1000);
                                car = ((GetNth(b, len_b-i) * GetNth(a, len_a-j)) + car)/1000;
                        }
                        for (int count=1; count<i; count++){
                                bpush(&t_1, 000);
                        }


                        if(car)
                                fpush(&t_1, car);

                        t_0 = add(t_0, t_1);
                 }

                 frre(t_1);
		
		
		return t_0;
	}
}


struct Node* parse(char input[]){

        int i = 0;
        int j = 0;

        int t[6] = { -1,-1,-1,-1,-1,-1};

	struct Node* res = NULL;
	struct Node* temp = NULL; 
        char previous_symbol = 0;

        if(input[0]=='=')
                exit(0);

        while(input[i]){
              	t[j] = converter(input[i]);
		

		if(input[i] == '='){
			frre(temp);
			return res;
		}
                
		if(t[j] != -1){
                        i++;
                        j++;
                        if(j>3){
                                printf("\n\t Invalid Input \t \n");
				free(res);
				free(temp);
				main();
                        }
			continue;
                }
                if(input[i] == ','){
                        bpush(&temp, calcnumber(t, j-1));
                        j = 0;
                }
                if(input[i] == '$'){
			bpush(&temp, calcnumber(t, j-1));
                        j = 0;
			
                        if(previous_symbol){

				if(previous_symbol=='+'){
					res = add(res, temp);
					previous_symbol = 0;
				}else{
					if(previous_symbol=='*'){
						res = mul(res, temp);
						previous_symbol = 0;
					}
				}
                        }else{
			
				res = temp;
				temp = NULL;
			}
                }

                if(input[i] == '+'){
                        previous_symbol = '+';
		}
                if(input[i] == '*'){
                        previous_symbol = '*';
		}

                i++;
        }
	return res;
}

int converter(char c)
{
        switch(c){
                case '0':
                        return 0;
                case '1':
                        return 1;
                case '2':
                        return 2;
                case '3':
                        return 3;
                case '4':
                        return 4;
                case '5':
                        return 5;
                case '6':
                        return 6;
                case '7':
                        return 7;
                case '8':
                        return 8;
                case '9':
                        return 9;
                default:
                        return -1;
        }
        return -1;
}


int calcnumber(int a[],int j){
        if(j == 0)
                return a[0];
        if(j == 1)
                return a[0]*10 + a[1];
        return a[0]*100 + a[1]*10 + a[2];
}


