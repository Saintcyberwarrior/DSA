void update_node(struct bst_node * a, int new_index, char word[21], struct lst_node* new_lst){
	struct bst_node*b = NULL;
	b = search(a, word);
	b->index = new_index;
	b->list = new_lst;
}

int search_node(struct bst_node*a, char word[21]){
	if(a->word == word)
		return 1;
	else if(a->word < word)
		search_node(a->right, word);
	else if(a->word > word)
		search_node(a->left, word);
	else if (a->word == NULL)
		return 0;
}

void reader(FILE *file){
	FILE * file = NULL;
	file = fopen("filename", "r");
	if(!file){
		fprintf(stderr, "Couldn't open File\n");
		return EXIT_FAILURE;
	}
	struct bst_node * Ultimate_tree = NULL;

	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	int line_no = 1;
	int i = 0;
	char chr;
	chr = getc(file); //input char form file
	while(chr != EOF){
		if(chr='\n'){
			line_no = line_no+1;
		}

		if(char == EndOfWord){
			if(search_node(Ultimate_tree, word)){
				update_node(Ultimate_tree,word,line_number);
			}else{
				bst_add(Ultimate_tree, NULL, word, NULL);
			}
			i = 0;
		}else{
			chr = word[i];
			i = i+1;
		}
	}
}

