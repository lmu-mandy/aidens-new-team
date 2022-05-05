Files:
	txt/ -- directory contains the text files
	tags.tsv -- contains the tags, see format below
	filtered_tags.tsv -- a filtered version of tags.csv where only NCBI-taxonomy-names are included (i.e. "species clues" such as "patient", "woman", etc. have been removed)

Columns in tags.tsv:
	species id
	document
	start: coordinate for the mention (number of characters into the file)
	end: coordinate for the mention (number of characters into the file)
	the actual text mention (i.e. substring between "start" and "end" in "document"
	category code (explained below)
	
Categories:
	A few special categories have been defined, useful for e.g. analysis of FNs
	
	0: misspelling
	1: OCR error (caused by OCR software when scanning articles into text format)
	2: incorrect name usage (e.g. using Drosophila when referring to D. melanogaster)
	20: wrong case (e.g. "drosophila melanogaster")
	3: enumeration (e.g. "V. vulnificus CMCP6 and YJ016", will result in two mentions)
	4: modifier (e.g. human in "human studies", mouse in "mouse gene", rat in "rat brain")
