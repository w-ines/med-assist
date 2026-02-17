// supabase/functions/healthcheck/index.ts
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// Désactiver la vérification JWT par défaut de Supabase
Deno.serve(async (req) => {
  // Vérifier le secret dans le header
  const authHeader = req.headers.get('x-healthcheck-secret')
  const expectedSecret = Deno.env.get('HEALTHCHECK_SECRET')
  
  if (authHeader !== expectedSecret) {
    return new Response('Unauthorized', { status: 401 })
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  )

  const { error } = await supabase
    .from('healthcheck')
    .update({ pinged_at: new Date().toISOString() })
    .eq('id', 1)

  if (error) {
    return new Response(JSON.stringify({ error }), { status: 500 })
  }

  return new Response(JSON.stringify({ status: 'ok', timestamp: new Date() }), {
    headers: { 'Content-Type': 'application/json' }
  })
})