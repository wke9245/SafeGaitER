# Dataset and attacker protocols

## Preprocessing

Sequences are sampled to 30 frames. Training uses independent temporal and silhouette augmentations. Evaluation uses fixed temporal sampling.

Noncanonical images are normalized by foreground height, centered horizontally by silhouette mass, and cropped or padded to 64 by 44. Canonical 64-by-44 silhouettes are binarized directly. Empty silhouettes are represented by zero-valued frames.

## CASIA-B

Subjects 001-074 train the target recognizer. Subjects 075-124 form the held-out recognition partition. The 11 views are 000, 018, 036, 054, 072, 090, 108, 126, 144, 162, and 180.

| Role | Subjects | Conditions | Sequences |
| --- | --- | --- | --- |
| Recognizer training | 001-074 | NM 01-04, BG 01, CL 01 | 4,884 |
| Reserved nonmembers | 001-074 | NM 05-06, BG 02, CL 02 | 3,256 |
| Recognition gallery | 075-124 | NM 01-04 | 2,200 |
| Recognition probes | 075-124 | NM 05-06, BG 01-02, CL 01-02 | 3,300 |

The held-out explanation, inversion, and audit pool includes all 5,500 sequences. Gallery queries are excluded from their own candidate set.

### Identity-matched membership

Attacker fitting uses subjects 001-049. Evaluation uses subjects 050-074. Each identity contributes 22 seed-fixed member NM sequences, all 11 BG 01 sequences, and all 11 CL 01 sequences. Reserved nonmembers provide the same view and condition proportions.

The fitting pool has 2,156 instances per class; the evaluation pool has 1,100 instances per class.

### Identity-disjoint membership

Fitting members come from subjects 001-049, and fitting nonmembers from subjects 075-099. Evaluation members come from subjects 050-074, and evaluation nonmembers from subjects 100-124.

Each pool has 2,750 instances per membership class. Fitting members are sampled without replacement. Evaluation members are sampled with replacement from 1,650 available target-training sequences. Reported sample counts distinguish instances from unique source sequences.

### Linkage and identity probing

Linkage fits on subjects 075-099 and evaluates on subjects 100-124, using 50,000 pairs per class for fitting and 20,000 pairs per class for evaluation.

Identity probing uses all 50 held-out identities with disjoint fitting and evaluation sequences for each identity.

## OU-MVLP

The official identity partition contains 5,153 training subjects and 5,154 held-out subjects. Sorted training IDs map to internal indices 0-5152, and sorted held-out IDs map to 5153-10306.

Target recognition training uses sequences 00 and 01 over all 14 views. Held-out sequence 01 is the gallery and sequence 00 is the probe.

Membership fitting uses member indices 0-2575 and nonmember indices 5153-7729. Evaluation uses member indices 2576-5152 and nonmember indices 7730-10306. Each membership class has 10,000 seed-fixed instances.

## Auxiliary adversarial pools

For CASIA-B target training, the auxiliary pool contains 4,884 OU-MVLP sequences: the first 407 sorted training subjects, both sequences, and views 000, 015, 030, 045, 060, and 090.

For OU-MVLP target training, the auxiliary pool is the 4,884-sequence CASIA-B recognition-training pool.

Auxiliary samples enter the adversarial privacy objective and are excluded from recognition losses and post-hoc membership labels.
