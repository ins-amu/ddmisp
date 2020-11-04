
import sys

import util

if sys.argv[1] == 'one':
    util.post.plot_solo_compact(int(sys.argv[2]), sys.argv[3], sys.argv[4].split(" "),
                                sys.argv[5].split(" "), sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
