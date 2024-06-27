$rs_file = "output_rs.txt"
$rs_phi = 0.07, 0.06, 0.05, 0.04
foreach ($phi in $rs_phi) {
    "phi: " + $phi >> $rs_file
    for ($i=1; $i -le 3; $i=$i+1 ) {
        "i: " + $i >> $rs_file
        $command = "python .\G007HW3.py 1000000 " + $phi + " 0.05 0.1 8886"
        Invoke-Expression $command >> $rs_file
        "" >> $rs_file
    }
    "--------------------------------------------------------------" >> $rs_file
}


$ss_file = "output_ss.txt"
$ss_epsilon = 0.06, 0.05, 0.04, 0.03
foreach ($epsilon in $ss_epsilon) {
    "epsilon: " + $epsilon >> $ss_file
    for ($i=1; $i -le 3; $i=$i+1 ) {
        "i: " + $i >> $ss_file
        $command = "python .\G007HW3.py 1000000 0.07 " + $epsilon + " 0.1 8888"
        Invoke-Expression $command >> $ss_file
        "" >> $ss_file
    }
    "--------------------------------------------------------------" >> $ss_file
}
