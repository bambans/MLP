all:
	make logic
	cat saidaPortasLogicas.txt
	make char
	cat saidaCaracteres.txt

char: characters.py dictionary.py dumpsChar
	make dumpsChar
	python characters.py > saidaCaracteres.txt

logic: logic.py dictionary.py dumpsLogic
	make dumpsLogic
	python logic.py > saidaPortasLogicas.txt

dumpsLogic:
	mkdir dumpsLogic
	mkdir matricesLogic

dumpsChar:
	mkdir dumpsChar
	mkdir matricesChar

clean:
	rm -f dumps*/*
	rmdir dumps*/
	rmdir matrices*/
	rm -f confusion_matrix_*
