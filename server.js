import Anthropic from '@anthropic-ai/sdk';
import express from 'express';
import rateLimit from 'express-rate-limit';
import CircuitBreaker from 'opossum';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

// ─── Rate Limiters ────────────────────────────────────────────────────────────

// Strict: 5 requests per minute per IP
const strictLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please wait a moment before trying again.' },
});

// Global: 30 requests per hour per IP
const hourlyLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Hourly limit reached. Please try again later.' },
});

// ─── In-Memory Cache ──────────────────────────────────────────────────────────

const cache = new Map();
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour
const MAX_CACHE_SIZE = 1000;

function cacheKey(ingredients) {
  return [...ingredients].sort().join('|').toLowerCase();
}

function fromCache(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL_MS) {
    cache.delete(key);
    return null;
  }
  return entry.data;
}

function toCache(key, data) {
  if (cache.size >= MAX_CACHE_SIZE) {
    // Evict oldest entry
    cache.delete(cache.keys().next().value);
  }
  cache.set(key, { data, ts: Date.now() });
}

// ─── Anthropic Client ─────────────────────────────────────────────────────────

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const SYSTEM_PROMPT = `You are an expert vegan chef and nutritionist. You suggest creative, delicious, 100% plant-based recipes.
All recipes must be completely vegan — no meat, fish, seafood, dairy (milk/cheese/butter/cream), eggs, or honey.
You respond only with valid JSON — no markdown fences, no explanation text.`;

async function fetchRecipesFromClaude(ingredients) {
  const userPrompt = `The user has these ingredients: ${ingredients.join(', ')}.

Return a JSON object with this exact structure:
{
  "exact_match_recipes": [
    {
      "name": "Recipe Name",
      "description": "1-2 sentence appetizing description",
      "ingredients": ["measured ingredient 1", "measured ingredient 2"],
      "instructions": ["Step 1.", "Step 2.", "Step 3."],
      "prep_time": "10 min",
      "cook_time": "20 min",
      "servings": 2,
      "difficulty": "Easy",
      "tags": ["quick", "high-protein"],
      "image_query": "vegan lentil soup bowl"
    }
  ],
  "related_recipes": [
    {
      "name": "Recipe Name",
      "description": "1-2 sentence description",
      "ingredients": ["measured ingredient 1", "measured ingredient 2"],
      "missing_ingredients": ["1 extra ingredient needed"],
      "instructions": ["Step 1.", "Step 2."],
      "prep_time": "5 min",
      "cook_time": "15 min",
      "servings": 2,
      "difficulty": "Easy",
      "tags": ["30-min", "one-pot"],
      "image_query": "vegan stir fry noodles"
    }
  ],
  "notes": "Any helpful notes about the ingredients, substitutions, or storage tips."
}

Rules:
- exact_match_recipes: recipes made with ONLY the provided ingredients (up to 5 recipes)
- related_recipes: recipes needing 1–3 extra common ingredients (up to 5 recipes)
- image_query: 3-5 words describing the dish for a photo search (e.g. "vegan buddha bowl")
- If ingredients are unusual or not typically used in cooking, still try your best and note this in "notes"
- Return ONLY the JSON object, nothing else`;

  const message = await anthropic.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 4096,
    system: SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userPrompt }],
  });

  const text = message.content[0].text.trim();
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) throw new Error('Claude returned non-JSON response');
  return JSON.parse(jsonMatch[0]);
}

// ─── Circuit Breaker ──────────────────────────────────────────────────────────

const breaker = new CircuitBreaker(fetchRecipesFromClaude, {
  timeout: 30000,                // 30s per request
  errorThresholdPercentage: 50,  // open after 50% failures
  resetTimeout: 60000,           // attempt recovery after 60s
  volumeThreshold: 3,            // need at least 3 calls before tripping
});

breaker.fallback(() => ({
  exact_match_recipes: [],
  related_recipes: [],
  notes: 'Our recipe service is temporarily unavailable due to high demand. Please try again in a minute.',
  _fallback: true,
}));

breaker.on('failure', (err) => console.error('[circuit failure]', err?.status, err?.message));
breaker.on('open', () => console.warn('[circuit] Opened — Claude API unreachable'));
breaker.on('halfOpen', () => console.info('[circuit] Half-open — testing recovery'));
breaker.on('close', () => console.info('[circuit] Closed — Claude API recovered'));

// ─── Image Fetching ───────────────────────────────────────────────────────────

// Simple in-memory image cache (query → URL)
const imageCache = new Map();

function queryHash(s) {
  let h = 0;
  for (const c of s) h = (Math.imul(31, h) + c.charCodeAt(0)) | 0;
  return Math.abs(h) % 10000;
}

async function fetchUnsplashImage(query) {
  if (!process.env.UNSPLASH_ACCESS_KEY) return null;
  try {
    const url = `https://api.unsplash.com/photos/random?query=${encodeURIComponent(query + ' vegan food')}&orientation=landscape&content_filter=high`;
    const res = await fetch(url, {
      headers: { Authorization: `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}` },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.urls?.small || null;
  } catch {
    return null;
  }
}

async function fetchPexelsImage(query) {
  if (!process.env.PEXELS_API_KEY) return null;
  try {
    const url = `https://api.pexels.com/v1/search?query=${encodeURIComponent(query + ' vegan food')}&per_page=1&orientation=landscape`;
    const res = await fetch(url, {
      headers: { Authorization: process.env.PEXELS_API_KEY },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.photos?.[0]?.src?.medium || null;
  } catch {
    return null;
  }
}

async function getImage(query) {
  if (imageCache.has(query)) return imageCache.get(query);

  let url =
    (await fetchUnsplashImage(query)) ||
    (await fetchPexelsImage(query)) ||
    `https://loremflickr.com/600/400/food?lock=${queryHash(query)}`;

  if (imageCache.size > 500) imageCache.delete(imageCache.keys().next().value);
  imageCache.set(query, url);
  return url;
}

async function addImages(recipes) {
  const allRecipes = [
    ...recipes.exact_match_recipes.map((r) => ({ ...r, _type: 'exact' })),
    ...recipes.related_recipes.map((r) => ({ ...r, _type: 'related' })),
  ];

  const withImages = await Promise.all(
    allRecipes.map(async (recipe) => ({
      ...recipe,
      image: await getImage(recipe.image_query || recipe.name),
    }))
  );

  return {
    ...recipes,
    exact_match_recipes: withImages.filter((r) => r._type === 'exact').map(({ _type, ...r }) => r),
    related_recipes: withImages.filter((r) => r._type === 'related').map(({ _type, ...r }) => r),
  };
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.post('/api/recipes', hourlyLimiter, strictLimiter, async (req, res) => {
  const { ingredients } = req.body;

  if (!Array.isArray(ingredients) || ingredients.length === 0) {
    return res.status(400).json({ error: 'Please provide at least one ingredient.' });
  }
  if (ingredients.length > 20) {
    return res.status(400).json({ error: 'Maximum 20 ingredients allowed.' });
  }

  const clean = ingredients
    .map((i) => String(i).trim().toLowerCase())
    .filter((i) => i.length > 0 && i.length <= 60);

  if (clean.length === 0) {
    return res.status(400).json({ error: 'No valid ingredients provided.' });
  }

  const key = cacheKey(clean);
  const cached = fromCache(key);
  if (cached) {
    return res.json({ ...cached, _cached: true });
  }

  try {
    const recipes = await breaker.fire(clean);
    const enriched = await addImages(recipes);

    if (!enriched._fallback) {
      toCache(key, enriched);
    }

    res.json(enriched);
  } catch (err) {
    console.error('[/api/recipes]', err.message);

    if (err.status === 429 || err.message?.includes('rate_limit')) {
      return res.status(429).json({ error: 'AI service rate limit reached. Please try again shortly.' });
    }
    if (err.message?.includes('invalid_api_key') || err.status === 401) {
      return res.status(500).json({ error: 'Server configuration error. Please contact support.' });
    }

    res.status(500).json({ error: 'Something went wrong. Please try again.' });
  }
});

// Debug endpoint — bypasses circuit breaker to show raw Claude error
app.get('/api/test', async (_req, res) => {
  try {
    const result = await fetchRecipesFromClaude(['tomato', 'garlic']);
    res.json({ ok: true, result });
  } catch (err) {
    res.status(500).json({
      ok: false,
      status: err.status,
      message: err.message,
      type: err.constructor?.name,
    });
  }
});

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    circuit: breaker.opened ? 'open' : breaker.halfOpen ? 'half-open' : 'closed',
    cache: { recipes: cache.size, images: imageCache.size },
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🌱 GreenPantry running → http://localhost:${PORT}`);
});
