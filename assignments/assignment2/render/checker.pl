#!/usr/bin/perl

use POSIX;
use Getopt::Std;

my @scene_names = ("rgb", "rgby", "rand10k", "rand100k", "biglittle", "littlebig", "pattern", "bouncingballs", "hypnosis", "fireworks", "snow", "snowsingle");
my @score_scene_names = ("rgb", "rand10k", "rand100k", "pattern", "snowsingle", "biglittle");

my $size = 1150;
my $renderMode = "cuda";

my %fast_times;

my $perf_points = 10;
my $min_perf_points = 1;
my $min_ratio = 0.1;
my $max_ratio = 5.0/6.0;

my $correctness_points = 2;

my %correct;

my %your_times;

sub usage {
    printf STDERR "$_[0]";
    printf STDERR "Usage: $0 [-h] [-R] [-s SIZE]\n";
    printf STDERR "    -h         Print this message\n";
    printf STDERR "    -R         Use reference (CPU-based) renderer\n";
    printf STDERR "    -s SIZE    Set image size\n";
    die "\n";
}

getopts('hRs:');
if ($opt_h) {
    usage();
}

if ($opt_s) {
    $size = int($opt_s);
}

if ($opt_R) {
    $renderMode = "ref";
}

`mkdir -p logs`;
`rm -rf logs/*`;

print "\n";
print ("--------------\n");
my $hostname = `hostname`;
chomp $hostname;
print ("Running tests on $hostname, size = $size, mode = $renderMode\n");
print ("--------------\n");

my $render_soln = "render_soln";
if (index(lc($hostname),"ghc") == -1) {
    $render_soln = "render_soln_latedays";
}

foreach my $scene (@scene_names) {
    print ("\nScene : $scene\n");
    my @sys_stdout = system ("./render -c $scene -s $size > ./logs/correctness_${scene}.log");
    my $return_value  = $?;
    if ($return_value == 0) {
        print ("Correctness passed!\n");
        $correct{$scene} = 1;
    }
    else {
        print ("Correctness failed ... Check ./logs/correctness_${scene}.log\n");
        $correct{$scene} = 0;
    }

    if (${scene} ~~ @score_scene_names) {
        my $your_time = `./render -r $renderMode --bench 0:4 $scene -s $size | tee ./logs/time_${scene}.log | grep Total:`;
        chomp($your_time);
        $your_time =~ s/^[^0-9]*//;
        $your_time =~ s/ ms.*//;

        print ("Your time : $your_time\n");
        $your_times{$scene} = $your_time;

        my $fast_time = `./$render_soln -r $renderMode --bench 0:4 $scene -s $size | tee ./logs/time_${scene}.log | grep Total:`;
        chomp($fast_time);
        $fast_time =~ s/^[^0-9]*//;
        $fast_time =~ s/ ms.*//;

        print ("Target Time: $fast_time\n");
        $fast_times{$scene} = $fast_time;
    }
}

print "\n";
print ("------------\n");
print ("Score table:\n");
print ("------------\n");

my $header = sprintf ("| %-15s | %-15s | %-15s | %-15s |\n", "Scene Name", "Target Time ", "Your Time", "Score");
my $dashes = $header;
$dashes =~ s/./-/g;
print $dashes;
print $header;
print $dashes;

my $total_score = 0;

foreach my $scene (@score_scene_names){
    my $score;
    my $your_time = $your_times{$scene};
    my $fast_time = $fast_times{$scene};

    if ($correct{$scene}) {
	$ratio = $fast_time/$your_time;
        if ($ratio >= $max_ratio) {
            $score = $perf_points + $correctness_points;
        }
        elsif ($ratio < $min_ratio) {
            $score = $correctness_points;
        }
        else {
            $score = $correctness_points + $min_perf_points
		+ floor(($perf_points-$min_perf_points)*($ratio-$min_ratio)/($max_ratio-$min_ratio));
        }
    }
    else {
        $your_time .= " (F)";
        $score = 0;
    }

    printf ("| %-15s | %-15s | %-15s | %-15s |\n", "$scene", "$fast_time", "$your_time", "$score");
    $total_score += $score;
}
print $dashes;
printf ("| %-15s   %-15s | %-15s | %-15s |\n", "", "", "Total score:",
    $total_score . "/" . ($perf_points+$correctness_points) * ($#score_scene_names + 1));
print $dashes;
