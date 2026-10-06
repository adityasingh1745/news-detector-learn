CREATE TABLE "analyses" (
	"id" serial PRIMARY KEY NOT NULL,
	"headline" text NOT NULL,
	"body" text,
	"verdict" text NOT NULL,
	"confidence" numeric(5, 2) NOT NULL,
	"score_real" numeric(5, 2) NOT NULL,
	"score_clickbait" numeric(5, 2) NOT NULL,
	"score_fake" numeric(5, 2) NOT NULL,
	"indicators" text[] DEFAULT '{}' NOT NULL,
	"model_used" text NOT NULL,
	"analyzed_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "feedback" (
	"id" serial PRIMARY KEY NOT NULL,
	"analysis_id" integer NOT NULL,
	"correct_label" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "feedback" ADD CONSTRAINT "feedback_analysis_id_analyses_id_fk" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id") ON DELETE no action ON UPDATE no action;