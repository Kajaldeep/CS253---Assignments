#! /bin/bash
echo -e "Enter two filenames : \c"
read input_file output_file

if [[ $output_file = '' ]]; then
    echo "Error !! Two filenames must be provided. " # error message displayed if only one filename is provided
    exit 0
else
    if [ -e $input_file ]; then # check whether input file exists or not
        echo "input file exists!"
        echo "Required columns are : " >$output_file
        awk -F "," '{print $1 " " $2 " " $3 " " $5 " " $6 " " $7 " " $10 " " $11}' $input_file >>$output_file
        echo"" >>$output_file
        echo "Name of the college whose HighestDegree is Bachelor’s : " >>$output_file
        awk -F "," '$3 == "Bachelor'"'"'s" {print $1}' $input_file >>$output_file
        echo >>$output_file
        echo "Geography: Average Admission Rate" >>$output_file
        export input_file
        export output_file
        awk -F ',' '{if (NR!=1) print $6}' $input_file | sort | uniq >geo.txt
        awk '{system(" ./avg.sh " $0)}' geo.txt
        echo"" >>$output_file
        echo "Top five colleges who have maximum MedianEarnings : " >>$output_file
        tail -n +2 $input_file | sort -nrk16 -t ',' | awk -F "," 'FNR < 6 {print $1 " " ":" " " $16}' >>$output_file
    else
        echo "Input file does not exist!!" # if input file does not exist
        exit 0
    fi
fi
