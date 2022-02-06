void parse(char input[]);
int converter(char);
int calcnumber(int a[],int j);

void
parse(char input[]){

	int i = 0;
	int j = 0;

	int t[6] = { -1,-1,-1,-1,-1,-1};
	int buf1;
	int buf2;
	char previous_symbol = 0;

	printf("%s\n",input);
	
	if(input[0]=='=')
		exit(0);

	while(input[i]){
		t[j] = converter(input[i]);
		printf("%d",t[j]);
		if(t[j] != -1){
			i++;
			j++;
			if(j>3){
				printf("\n\nCyuka Blyat\n\n");
				return;
			}
			continue;
		}
		if(input[i] == ','){
			printf("\n%d\n",calcnumber(t,j-1));
			j = 0;
			printf("Call the push function\n");
		}
		if(input[i] == '$'){
			printf("\n%d\n",calcnumber(t,j-1));
			j = 0;
			printf("Call the push function for the last time\n");
			printf("One linked list done\n");

			if(previous_symbol){
				printf(
				"Call the (%c) function with last two lists and store the sum in first list.\n",
				previous_symbol);
			}
		}

		if(input[i] == '+')
			previous_symbol = '+';
		if(input[i] == '*')
			previous_symbol = '*';
		if(input[i] == '/')
			previous_symbol = '/';
		if(input[i] == '-')
			previous_symbol = '-';


		i++;
	}






}

int
converter(char c)
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


int
calcnumber(int a[],int j){
	if(j == 0)
		return a[0];
	if(j == 1)
		return a[0]*10 + a[1];
	return a[0]*100 + a[1]*10 + a[2];
}

