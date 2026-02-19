import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

Deno.serve(async (req) => {
  const authHeader = req.headers.get("x-healthcheck-secret")
  const expectedSecret = Deno.env.get("HEALTHCHECK_SECRET")

  if (!expectedSecret || authHeader !== expectedSecret) {
    return new Response("Unauthorized", { status: 401 })
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL")
  const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")

  if (!supabaseUrl || !serviceRoleKey) {
    return new Response("Missing Supabase env vars", { status: 500 })
  }

  const supabase = createClient(supabaseUrl, serviceRoleKey)

  const { error } = await supabase
    .from("healthcheck")
    .update({ pinged_at: new Date().toISOString() })
    .eq("id", 1)

  if (error) {
    return new Response(JSON.stringify({
      message: error.message,
      details: error.details,
      hint: error.hint,
      code: error.code,
    }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }

  return new Response(JSON.stringify({ status: "ok", timestamp: new Date() }), {
    headers: { "Content-Type": "application/json" },
  })
})
