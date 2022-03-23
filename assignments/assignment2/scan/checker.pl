#!/usr/bin/perl

use POSIX;
use Getopt::Std;


my @element_counts = ("10000", "100000", "1000000", "2000000");

my %fast_times; 
my %your_times; 

my $perf_points = 1.25;
my %correct;
my $mode = 'scan';
my $input = 'random';
my $thrust_flag = '';

`mkdir -p logs`;
`rm -rf logs/*`;
`mkdir logs/test`;
`mkdir logs/soln`;

sub usage {
    printf STDERR "$_[0]";
    printf STDERR "Usage: $0 [-h] [-t] [-m MODE] [-i INPUT]\n";
    printf STDERR "    -h         Print this message\n";
    printf STDERR "    -t         Use thrust implementation as performance target\n";
    printf STDERR "    -m MODE    Specify mode ('scan'";
    printf STDERR " or 'find_peaks')\n";
    printf STDERR "    -i INPUT   Specify input (either 'test1' or 'random'\n";
    die "\n";
}

getopts('htm:i:');
if ($opt_h) {
    usage();
}

if ($opt_t) {
    $thrust_flag =  '-t';
}

if ($opt_m) {
    $mode = $opt_m;
}

if ($opt_i) {
    $input = $opt_i;
}

print("Mode: $mode\n"); 
print("Input: $input\n");
if (!($thrust_flag eq '')) {
    print("Comparing to thrust\n");
}

print "\n";
print ("--------------\n");
print ("Running tests:\n");
print ("--------------\n");

foreach my $element_count (@element_counts) {
    print ("\nElement Count: $element_count\n");
    my @sys_stdout = system ("./cudaScan -m ${mode} -i $input -n $element_count > ./logs/test/${mode}_correctness_${element_count}.log");
    my $return_value  = $?;
    if ($return_value == 0) {
        print ("Correctness passed!\n");
        $correct{$element_count} = 1;
    }
    else {
        print ("Correctness failed\n");
        $correct{$scene} = 0;
    }

    my $your_time = `./cudaScan -m ${mode} -i $input -n $element_count | tee ./logs/test/${mode}_time_${element_count}.log | grep GPU_time:`;
    chomp($your_time);
    $your_time =~ s/^[^0-9]*//;
    $your_time =~ s/ ms.*//;
    print ("Your Time: $your_time\n"); 
    
    my $fast_time = `./cudaScan_soln -m ${mode} -i $input -n $element_count $thrust_flag | tee ./logs/soln/${mode}_time_${element_count}.log | grep GPU_time:`;
    chomp($fast_time);
    $fast_time =~ s/^[^0-9]*//;
    $fast_time =~ s/ ms.*//;
    print ("Target Time: $fast_time\n"); 

    $your_times{$element_count} = $your_time;
    $fast_times{$element_count} = $fast_time;
}

print "\n";
print ("-------------------------\n");
print (ucfirst($mode). " Score Table:\n");
print ("-------------------------\n");

my $header = sprintf ("| %-15s | %-15s | %-15s | %-15s |\n", "Element Count", "Target Time", "Your Time", "Score");
my $dashes = $header;
$dashes =~ s/./-/g;
print $dashes;
print $header;
print $dashes;

my $total_score = 0;

foreach my $element_count (@element_counts){
    my $score;
    my $fast_time = $fast_times{$element_count};
    my $time = $your_times{$element_count};

    if ($correct{$element_count}) {
        if ($time <= 1.20 * $fast_time) {
            $score = $perf_points;
        }
        else {
            $score = $perf_points * 1.20 * ($fast_time /$time);
        }
    }
    else {
        $time .= " (F)";
        $score = 0;
    }

    printf ("| %-15s | %-15s | %-15s | %-15s |\n", "$element_count", "$fast_time", "$time", "$score");
    $total_score += $score;
}
print $dashes;
printf ("| %-33s | %-15s | %-15s |\n", "", "Total score:", 
    $total_score . "/" . ($perf_points * keys %fast_times));
print $dashes;
