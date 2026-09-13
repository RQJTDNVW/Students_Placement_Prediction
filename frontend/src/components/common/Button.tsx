import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { LoaderCircle } from 'lucide-react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'text' | 'danger' | 'icon'
  size?: 'sm' | 'md'
  loading?: boolean
  children?: ReactNode
}

export function Button({ variant = 'secondary', size = 'md', loading = false, className = '', children, disabled, ...props }: ButtonProps) {
  const variants = {
    primary: 'primary-action border border-cyan bg-cyan text-page hover:bg-cyan/85',
    secondary: 'border border-line bg-transparent text-ink hover:border-cyan/70 hover:bg-raised/60',
    text: 'border border-transparent text-muted underline-offset-4 hover:text-cyan hover:underline',
    danger: 'border border-negative/30 bg-negative/10 text-negative hover:bg-negative/15',
    icon: 'border border-line bg-transparent text-muted hover:border-cyan/70 hover:bg-raised/60 hover:text-cyan',
  }
  const sizes = { sm: 'min-h-8 px-3 text-xs', md: 'min-h-10 px-4 text-sm' }
  return <button type="button" className={`inline-flex items-center justify-center gap-2 rounded-[7px] font-semibold transition-colors duration-150 disabled:opacity-45 ${variants[variant]} ${variant === 'icon' ? 'h-9 w-9 p-0' : sizes[size]} ${className}`} disabled={disabled || loading} {...props}>
    {loading && <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin" />}
    {children}
  </button>
}
