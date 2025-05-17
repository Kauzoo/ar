while true; do
    read -p "Do you wish to perform this action?" yesno
    case $yesno in
        [Yy]* ) 
            echo "You answered yes"
        ;;
        [Nn]* ) 
            echo "You answered no, exiting"
            exit
        ;;
        * ) echo "Answer either yes or no!";;
    esac
done

# Detect OS-Version
cat /etc/os-release | grep "ID_LIKE="