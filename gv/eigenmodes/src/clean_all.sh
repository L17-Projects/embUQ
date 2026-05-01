#!/bin/bash

del_all() {
	rm -r parameter/* 2>/dev/null
	rm -r trj_eq/* 2>/dev/null
	rm -r stats/* 2>/dev/null
	rm -r restart/* 2>/dev/null
	rm -r logs/* 2>/dev/null
	rm -r h5/* 2>/dev/null
}

cases () {
	case $1 in 
		[yY]) 
		printf "\n"
		del_all
		printf "\nCleaned!\n" ;;
		[nN])
		printf "\nExiting without cleaning\n" ;;
		*)
		printf "\nInvalid option '$1'. Use y, Y, n or N.\n" ;;
	esac
}

read -n 1 -p $'Are you sure you want to clean all? [yY/nN] \n' reply; 

cases $reply

#case ${reply1} in 
#    [nN])
#    printf "\nExiting without cleaning\n" ;;
#    [yY]) 
#    read -n 1 -p $'\nReally? [yY/nN] \n' reply2; 
#    cases $reply2
#    ;;
#    *)
#    printf "\nInvalid option '$reply1'. Use y, Y, n or N.\n" ;;
#esac
