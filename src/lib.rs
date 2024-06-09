use std::borrow::Borrow;

use p3_air::{ Air, AirBuilder, AirBuilderWithPublicValues, BaseAir };
use p3_baby_bear::{ BabyBear, DiffusionMatrixBabyBear };
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ AbstractField, Field, PrimeField64 };
use p3_fri::{ FriConfig, TwoAdicFriPcs };
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{ Poseidon2, Poseidon2ExternalMatrixGeneral };
use p3_symmetric::{ PaddingFreeSponge, TruncatedPermutation };
use p3_uni_stark::{ prove, verify, StarkConfig };
use rand::thread_rng;

//TODO: Add constrain to the coefficient of the polynomial
//TODO: Figure out how to add constrain between the public inputs and row `value` during transition phase

pub struct UniVariAir {
    pub degree: usize,
}

impl<F> BaseAir<F> for UniVariAir {
    fn width(&self) -> usize {
        NUM_UNIVAR_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for UniVariAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let public_inputs = (0..self.degree + 2).map(|i| pis[i]).collect::<Vec<_>>();

        let (cur, next) = (main.row_slice(0), main.row_slice(1));
        let cur: &UniVariRow<AB::Var> = (*cur).borrow();
        let next: &UniVariRow<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_eq(cur.pow_x, AB::Expr::one());

        builder.when_first_row().assert_eq(next.pow_x, public_inputs[0]);
        builder.when_first_row().assert_eq(cur.pow_x * cur.coeff, cur.product);

        let mut when_transition = builder.when_transition();

        when_transition.assert_eq(next.pow_x * next.coeff + cur.product, next.product);

        builder.when_last_row().assert_eq(cur.product, public_inputs[self.degree + 1]);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(
    x: u32,
    n: usize,
    coeff: Vec<u32>
) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    assert_eq!(n, coeff.len());

    let mut trace = RowMajorMatrix::new(vec![F::zero(); n * NUM_UNIVAR_COLS], NUM_UNIVAR_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<UniVariRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = UniVariRow::new(
        F::from_canonical_u32(1),
        F::from_canonical_u32(coeff[0]),
        F::from_canonical_u32(1 * coeff[0])
    );

    for i in 1..coeff.len() {
        rows[i].pow_x = rows[i - 1].pow_x * F::from_canonical_u32(x);
        rows[i].coeff = F::from_canonical_u32(coeff[i]);
        rows[i].product = rows[i].pow_x * rows[i].coeff + rows[i - 1].product;
    }

    for i in 0..n {
        let j = i * NUM_UNIVAR_COLS;
        println!("{:?}, {:?}, {:?}", trace.values[j], trace.values[j + 1], trace.values[j + 2]);
    }

    trace
}

const NUM_UNIVAR_COLS: usize = 3;

pub struct UniVariRow<F> {
    pub pow_x: F,
    pub coeff: F,
    pub product: F,
}

impl<F> UniVariRow<F> {
    const fn new(pow_x: F, coeff: F, product: F) -> UniVariRow<F> {
        UniVariRow { pow_x, coeff, product }
    }
}

impl<F> Borrow<UniVariRow<F>> for [F] {
    fn borrow(&self) -> &UniVariRow<F> {
        debug_assert_eq!(self.len(), NUM_UNIVAR_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<UniVariRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

type Val = BabyBear;
type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = FieldMerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    8
>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

#[test]
fn test_public_value() {
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut thread_rng()
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};

    //TODO: Use HashMap to map the coefficient and coefficient-idx of the polynomial
    let coeff = (0..16).map(|i| i).collect::<Vec<u32>>();

    let trace = generate_trace_rows::<Val>(2, 1 << 4, coeff.clone());
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 28,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm.clone());

    let mut pis = Vec::<BabyBear>::new();
    //push x =2 as public input
    pis.push(BabyBear::from_canonical_u32(2));
    //push poly coefficient

    for var in coeff.clone() {
        pis.push(BabyBear::from_canonical_u32(var));
    }
    //push poly eval at x = 2
    pis.push(BabyBear::from_canonical_u32(917506));

    println!("pis length = {:#?}", pis);

    let circuit = UniVariAir {
        degree: coeff.len(),
    };
    let proof = prove(&config, &circuit, &mut challenger, trace, &pis);
    let mut challenger = Challenger::new(perm);
    verify(&config, &circuit, &mut challenger, &proof, &pis).expect("verification failed");
}