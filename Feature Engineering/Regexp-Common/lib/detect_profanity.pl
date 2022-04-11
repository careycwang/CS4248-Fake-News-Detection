use Regexp::Common qw /profanity/;

sub profanity_
{   
    my ($a) = @_;
    if ($RE{profanity}->matches($a)) {
        return 1;
    } else{ 
        return 0
    }
}

1;
