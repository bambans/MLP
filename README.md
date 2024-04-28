<!--
ACH2016-102 - Inteligência Artificial
Profa. Dra. Sarajane Marques Peres

Alunos - NUSP:
    Ezequiel Park - 5172519
    Otávio Rodrigues Bambans - 12701582
    Vitor Ferreira Sacchi - 12542776

Implementação do MLP para o reconhecimento dos caracteres da Fausett e para a resolução do problema das portas lógicas.
 -->

# MLP for Character recognition and Logic Ports

- It is possible to use `make` to run the project, so do it:
    - `make clean` to clear all dumps (Be careful to use this command);
    - `make` to run MLP for both Character Recognition and Logic Ports;
    - `make char` to run MLP for Character Recognition;
    - `make logic` to run MLP for the Logic Ports problem;
- After running, a file called `saida*.txt` will be generated, where `*` stands for "Caracteres" for the Character Recognition problem and "PortasLogicas" for Logic Ports problem.
- After running, more specificly the tests, an image called `confusion_matrix_*.png` will be created, in which `*` stands for the name of the input file.
