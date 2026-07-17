import Link from 'next/link';
import { gitConfig } from '@/lib/shared';

const CONFIG_FIELDS = [
  { name: 'vocab_size', type: 'int', value: '32768' },
  { name: 'n_layer', type: 'int', value: '26' },
  { name: 'n_head', type: 'int', value: '20' },
  { name: 'n_kv_head', type: 'int', value: '4' },
  { name: 'd_model', type: 'int', value: '1280' },
  { name: 'd_ff', type: 'int', value: '3456' },
  { name: 'seq_len', type: 'int', value: '2048' },
  { name: 'rope_theta', type: 'float', value: '10000.0' },
  { name: 'norm_eps', type: 'float', value: '1e-6' },
];

const SPECS = [
  { value: '~10B tokens', detail: 'FineWeb-Edu, ≈Chinchilla-optimal' },
  { value: 'Muon + AdamW', detail: 'WSD schedule, bf16 autocast' },
  { value: '≈20 h · ~$50', detail: 'the real run, on one H100' },
  { value: '100 tests', detail: 'no transformers dependency' },
];

const COMPONENTS = [
  {
    name: 'RMSNorm',
    detail: 'Pre-norm, fp32 compute cast back',
    href: '/docs/architecture/rmsnorm',
    file: 'tinyllm/model.py',
  },
  {
    name: 'RoPE',
    detail: 'Rotary positions, NeoX half-split, precomputed tables',
    href: '/docs/architecture/rope',
    file: 'tinyllm/model.py',
  },
  {
    name: 'Attention',
    detail: 'Grouped-query (20 Q / 4 KV heads) with QK-norm, causal SDPA',
    href: '/docs/architecture/attention',
    file: 'tinyllm/model.py',
  },
  {
    name: 'SwiGLU',
    detail: 'Gated MLP, d_ff 3456, no biases anywhere',
    href: '/docs/architecture/swiglu',
    file: 'tinyllm/model.py',
  },
  {
    name: 'Embeddings & init',
    detail: 'Tied input/output, 1/√(2L) residual scaling',
    href: '/docs/architecture/embeddings-and-init',
    file: 'tinyllm/model.py',
  },
  {
    name: 'Tokenizer',
    detail: 'Byte-level BPE trained on 250 MB of FineWeb-Edu',
    href: '/docs/tokenizer',
    file: 'tinyllm/tokenizer.py',
  },
  {
    name: 'Data pipeline',
    detail: 'uint16 shards, memmapped random-crop sampler',
    href: '/docs/data',
    file: 'tinyllm/data.py',
  },
  {
    name: 'Training loop',
    detail: 'Grad accumulation, bf16 autocast, WSD schedule',
    href: '/docs/training',
    file: 'tinyllm/train.py',
  },
  {
    name: 'Muon',
    detail: 'Newton-Schulz orthogonalized momentum',
    href: '/docs/training/muon',
    file: 'tinyllm/muon.py',
  },
  {
    name: 'Distributed',
    detail: 'Plain DDP via torchrun, no_sync accumulation',
    href: '/docs/training/distributed',
    file: 'tinyllm/train.py',
  },
  {
    name: 'Checkpointing',
    detail: 'Atomic writes, bit-exact resume',
    href: '/docs/training/checkpointing',
    file: 'tinyllm/train.py',
  },
  {
    name: 'Evaluation',
    detail: 'HellaSwag harness on the pretrained checkpoint',
    href: '/docs/evaluation',
    file: 'tinyllm/eval_hellaswag.py',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      {/* Hero */}
      <section className="mx-auto w-full max-w-6xl px-6 pt-16 pb-14 md:pt-24">
        <div className="grid items-center gap-12 lg:grid-cols-[1.2fr_1fr]">
          <div>
            <p className="mb-5 font-mono text-xs tracking-widest text-fd-muted-foreground uppercase">
              Pretrained from scratch · no transformers dependency
            </p>
            <h1 className="text-4xl font-semibold tracking-tight text-balance md:text-5xl">
              <span className="font-mono tabular-nums">489,297,408</span>{' '}
              parameters. No pre-built components.
            </h1>
            <p className="mt-6 max-w-prose text-fd-muted-foreground">
              tiny-llm is a decoder-only transformer pretrained on ~10B tokens
              of FineWeb-Edu. The model, the byte-level BPE tokenizer, the data
              pipeline, the Muon + AdamW optimizer, and the training loop are
              all written from scratch in PyTorch;{' '}
              <code className="font-mono text-sm">nn.Linear</code> and SDPA are
              the lowest-level primitives used. Developed and verified on a
              laptop; the real run costs about $50 on one H100.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                href="/docs"
                className="rounded-full bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-colors hover:bg-fd-primary/85 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring"
              >
                Read the docs
              </Link>
              <Link
                href="/docs/getting-started"
                className="rounded-full border border-fd-border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring"
              >
                Getting started
              </Link>
            </div>
          </div>

          {/* The config that defines the model, verbatim from the repo */}
          <figure className="overflow-hidden rounded-lg border border-fd-border bg-fd-card">
            <figcaption className="flex items-center justify-between border-b border-fd-border px-4 py-2.5 font-mono text-xs text-fd-muted-foreground">
              tinyllm/config.py
              <a
                href={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/tinyllm/config.py`}
                className="underline underline-offset-4 hover:text-fd-foreground"
              >
                view source
              </a>
            </figcaption>
            <pre className="overflow-x-auto px-4 py-4 font-mono text-[13px] leading-6">
              <code>
                <span className="text-fd-muted-foreground">@dataclass</span>
                {'\n'}
                <span className="text-fd-muted-foreground">class</span>{' '}
                ModelConfig:{'\n'}
                {CONFIG_FIELDS.map((f) => (
                  <span key={f.name}>
                    {'    '}
                    <span className="text-fd-muted-foreground">
                      {f.name}: {f.type} ={' '}
                    </span>
                    <span className="font-medium">{f.value}</span>
                    {'\n'}
                  </span>
                ))}
                {'\n'}
                <span className="text-fd-muted-foreground">
                  # defaults = the &quot;d26&quot; preset
                </span>
                {'\n'}
                <span className="text-fd-muted-foreground">
                  # → 489,297,408 params
                </span>
              </code>
            </pre>
          </figure>
        </div>
      </section>

      {/* Spec strip */}
      <section className="border-y border-fd-border">
        <dl className="mx-auto grid w-full max-w-6xl grid-cols-2 gap-x-8 gap-y-6 px-6 py-8 lg:grid-cols-4">
          {SPECS.map((s) => (
            <div key={s.value}>
              <dt className="sr-only">{s.detail}</dt>
              <dd className="font-mono text-sm font-medium">{s.value}</dd>
              <dd className="mt-1 text-sm text-fd-muted-foreground">
                {s.detail}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      {/* Component index */}
      <section className="mx-auto w-full max-w-6xl px-6 py-14">
        <h2 className="text-xl font-semibold tracking-tight">
          What is implemented here
        </h2>
        <p className="mt-2 max-w-prose text-sm text-fd-muted-foreground">
          Each page covers one component: what it is, the math, the actual code
          lifted from the repo, and why that choice over the alternatives.
        </p>
        <ul className="mt-8 grid gap-x-8 sm:grid-cols-2 lg:grid-cols-3">
          {COMPONENTS.map((c) => (
            <li key={c.name} className="border-t border-fd-border">
              <Link
                href={c.href}
                className="group block py-4 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring"
              >
                <span className="flex items-baseline justify-between gap-3">
                  <span className="font-medium group-hover:underline group-hover:underline-offset-4">
                    {c.name}
                  </span>
                  <span className="shrink-0 font-mono text-xs text-fd-muted-foreground">
                    {c.file}
                  </span>
                </span>
                <span className="mt-1 block text-sm text-fd-muted-foreground">
                  {c.detail}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Closing band */}
      <section className="border-t border-fd-border">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-6 py-10 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-prose text-sm text-fd-muted-foreground">
            The whole pipeline runs on a laptop: the 13.1M-parameter{' '}
            <code className="font-mono">smoke</code> preset exercises the
            identical code path in about 15 minutes on Apple Silicon.
          </p>
          <Link
            href="/docs/runbooks/local-smoke"
            className="shrink-0 text-sm font-medium underline underline-offset-4 hover:text-fd-muted-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring"
          >
            Run the local smoke test →
          </Link>
        </div>
      </section>
    </main>
  );
}
