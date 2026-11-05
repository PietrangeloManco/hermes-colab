// hermes-colab Desktop page: your Colab runs, live. Plain ESM (no build); imports only the SDK and React.
import {
  Badge,
  Button,
  Codicon,
  EmptyState,
  LogView,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  StatusDot,
  queryClient,
  useQuery
} from '@hermes/plugin-sdk'
import { useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'hermes-colab'
const PATH = '/colab'
const ACTIVE = new Set(['queued', 'waiting_gpu', 'assigning', 'uploading', 'starting', 'running', 'resuming', 'cancelling'])
const TONE = { succeeded: 'good', failed: 'bad', cancelled: 'muted', waiting_gpu: 'warn', resuming: 'warn' }
const BADGE = { succeeded: 'success', failed: 'destructive', cancelled: 'muted', waiting_gpu: 'warn', resuming: 'warn' }

let rest = null

const muted = 'text-(--ui-text-tertiary)'

function Meter({ fraction }) {
  const pct = Math.max(0, Math.min(1, fraction || 0)) * 100
  return jsx('div', {
    className: 'h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-quaternary)',
    role: 'progressbar',
    'aria-valuenow': Math.round(pct),
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    children: jsx('div', { className: 'h-full rounded-full bg-(--ui-accent) transition-[width]', style: { width: `${pct}%` } })
  })
}

function fraction(progress) {
  if (!progress) return 0
  if (typeof progress.fraction === 'number') return progress.fraction
  if (progress.total) return (progress.step || 0) / progress.total
  return 0
}

function Stat({ label, value }) {
  return jsxs('div', {
    className: 'min-w-0',
    children: [
      jsx('div', { className: `text-[0.6875rem] uppercase tracking-wide ${muted}`, children: label }),
      jsx('div', { className: 'truncate text-sm tabular-nums', title: String(value ?? '—'), children: value ?? '—' })
    ]
  })
}

function RunRow({ run, selected, onSelect }) {
  const active = ACTIVE.has(run.state)
  return jsxs('button', {
    type: 'button',
    onClick: onSelect,
    className:
      'flex w-full min-w-0 flex-col gap-1.5 rounded-md px-3 py-2 text-left hover:bg-(--chrome-action-hover) ' +
      (selected ? 'bg-(--ui-bg-quaternary)' : ''),
    children: [
      jsxs('div', {
        className: 'flex min-w-0 items-center gap-2',
        children: [
          jsx(StatusDot, { tone: TONE[run.state] || (active ? 'good' : 'muted'), className: active ? 'animate-pulse' : '' }),
          jsx('span', { className: 'min-w-0 flex-1 truncate text-sm font-medium', title: run.name, children: run.name }),
          jsx(Badge, { variant: 'outline', children: run.accelerator })
        ]
      }),
      jsxs('div', {
        className: `flex min-w-0 items-center gap-2 text-xs ${muted}`,
        children: [
          jsx('span', { className: 'shrink-0', children: run.state_label }),
          jsx('span', { className: 'min-w-0 flex-1 truncate', title: run.progress?.label || run.message || '', children: run.progress?.label || run.message || '' }),
          jsx('span', { className: 'shrink-0 tabular-nums', children: `${run.elapsed_minutes} min` })
        ]
      }),
      active ? jsx(Meter, { fraction: fraction(run.progress) }) : null
    ]
  })
}

function Detail({ id }) {
  const { data, error } = useQuery({
    queryKey: [ID, 'run', id],
    queryFn: () => rest(`/runs/${encodeURIComponent(id)}`),
    refetchInterval: q => (q.state.data && !ACTIVE.has(q.state.data.run?.state) ? false : 4000)
  })
  const [busy, setBusy] = useState(false)
  if (error) return jsx(EmptyState, { title: 'Could not load this run', description: String(error.message || error) })
  const run = data?.run
  if (!run) return null
  const active = ACTIVE.has(run.state)
  const act = async verb => {
    setBusy(true)
    try {
      await rest(`/runs/${encodeURIComponent(run.id)}/${verb}`, { method: 'POST' })
    } finally {
      setBusy(false)
      queryClient.invalidateQueries({ queryKey: [ID] })
    }
  }
  const gpu = run.gpu
  return jsxs('div', {
    className: 'flex min-h-0 min-w-0 flex-1 flex-col gap-4 p-4',
    children: [
      jsxs('div', {
        className: 'flex min-w-0 flex-wrap items-center gap-2',
        children: [
          jsx('h2', { className: 'min-w-0 flex-1 truncate text-base font-semibold', title: run.name, children: run.name }),
          jsx(Badge, { variant: BADGE[run.state] || 'default', children: run.state_label }),
          active
            ? jsx(Button, { size: 'sm', variant: 'outline', disabled: busy || run.state === 'cancelling', onClick: () => act('cancel'),
                children: [jsx(Codicon, { name: 'debug-stop' }), ' Cancel'] })
            : null,
          run.state === 'failed' || run.state === 'cancelled'
            ? jsx(Button, { size: 'sm', variant: 'secondary', disabled: busy, onClick: () => act('resume'),
                children: [jsx(Codicon, { name: 'debug-restart' }), ' Resume on a fresh runtime'] })
            : null
        ]
      }),
      active || run.progress
        ? jsxs('div', {
            className: 'flex flex-col gap-1.5',
            children: [
              jsx(Meter, { fraction: run.state === 'succeeded' ? 1 : fraction(run.progress) }),
              jsx('div', { className: `truncate text-xs ${muted}`, children: run.progress?.label || run.message || '' })
            ]
          })
        : null,
      jsxs('div', {
        className: 'grid grid-cols-[repeat(auto-fit,minmax(8.5rem,1fr))] gap-3',
        children: [
          jsx(Stat, { label: 'Runtime', value: gpu?.name || run.accelerator }),
          jsx(Stat, { label: 'GPU use', value: gpu ? `${gpu.util}% · ${(gpu.mem_used_mb / 1024).toFixed(1)}/${(gpu.mem_total_mb / 1024).toFixed(0)} GB` : null }),
          jsx(Stat, { label: 'Elapsed', value: `${run.elapsed_minutes} min` }),
          jsx(Stat, { label: 'Compute units', value: `≈ ${run.compute_units_estimate}` }),
          jsx(Stat, { label: 'Attempt', value: run.resumes ? `${run.attempt} (${run.resumes} resumed)` : run.attempt })
        ]
      }),
      run.error && run.state !== 'succeeded'
        ? jsx('div', { className: 'rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive break-words', children: run.error })
        : null,
      jsxs('div', {
        className: 'grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]',
        children: [
          jsxs('section', {
            className: 'flex min-h-48 min-w-0 flex-col gap-1.5',
            children: [
              jsx('div', { className: `text-xs font-medium ${muted}`, children: 'Log' }),
              jsx(LogView, {
                className: 'min-h-40 flex-1 overflow-auto whitespace-pre-wrap break-all',
                children: (run.log || []).join('\n') || 'No output yet.'
              })
            ]
          }),
          jsxs('section', {
            className: 'flex min-w-0 flex-col gap-3',
            children: [
              jsxs('div', {
                className: 'flex min-w-0 flex-col gap-1',
                children: [
                  jsx('div', { className: `text-xs font-medium ${muted}`, children: `Outputs (${run.outputs?.length || 0})` }),
                  ...(run.outputs || []).map(name =>
                    jsx('div', { key: name, className: 'truncate font-mono text-xs', title: name, children: name })),
                  jsx('div', { className: `break-all text-[0.6875rem] ${muted}`, children: `${run.local_dir}/outputs` })
                ]
              }),
              jsxs('div', {
                className: 'flex min-w-0 flex-col gap-1',
                children: [
                  jsx('div', { className: `text-xs font-medium ${muted}`, children: 'Events' }),
                  ...(run.events || []).slice().reverse().map((e, i) =>
                    jsx('div', { key: i, className: 'break-words text-xs', children: e }))
                ]
              })
            ]
          })
        ]
      })
    ]
  })
}

function ColabPage() {
  const { data, error, isLoading } = useQuery({
    queryKey: [ID, 'runs'],
    queryFn: () => rest('/runs'),
    refetchInterval: q => ((q.state.data?.runs || []).some(r => ACTIVE.has(r.state)) ? 4000 : 20000)
  })
  const [picked, setPicked] = useState(null)
  const runs = data?.runs || []
  const selected = picked && runs.some(r => r.id === picked) ? picked : runs[0]?.id
  if (error)
    return jsx(EmptyState, {
      title: 'hermes-colab backend unavailable',
      description: 'Enable the hermes-colab plugin for this profile and restart the gateway.'
    })
  if (!isLoading && !runs.length)
    return jsx(EmptyState, {
      title: 'No Colab runs yet',
      description: 'Ask Hermes to run a script or notebook on a Colab GPU, e.g. “train train.py on a T4”.'
    })
  return jsxs('div', {
    className: 'flex h-full min-h-0 w-full min-w-0 flex-col md:flex-row',
    children: [
      jsxs('aside', {
        className: 'flex max-h-64 min-w-0 shrink-0 flex-col gap-1 overflow-auto border-b border-(--ui-stroke-secondary) p-2 md:max-h-none md:w-72 md:border-r md:border-b-0',
        children: [
          jsx('div', { className: `px-3 pt-1 pb-1 text-xs font-medium ${muted}`, children: 'Colab runs' }),
          ...runs.map(run => jsx(RunRow, { key: run.id, run, selected: run.id === selected, onSelect: () => setPicked(run.id) }))
        ]
      }),
      selected ? jsx(Detail, { id: selected }) : null
    ]
  })
}

export default {
  id: ID,
  name: 'Colab',
  register(ctx) {
    rest = ctx.rest
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: PATH }, render: () => jsx(ColabPage, {}) },
      { id: 'nav', area: SIDEBAR_NAV_AREA, data: { path: PATH, label: 'Colab', codicon: 'rocket' } }
    ])
    ctx.onEvent('plugin.hermes-colab.run.finished', () => queryClient.invalidateQueries({ queryKey: [ID] }))
  }
}
